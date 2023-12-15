/*
 * audio_processing.c
 *
 *  Created on: May 17, 2021
 *      Author: sydxrey
 *
 *
 * === Audio latency ===
 *
 * Receive DMA copies audio samples from the CODEC to the DMA buffer (through the I2S serial bus) as interleaved stereo samples
 * (either slots 0 and 2 for the violet on-board "LineIn" connector, or slots 1 and 3, for the pair of on-boards microphones).
 *
 * Transmit DMA copies audio samples from the DMA buffer to the CODEC again as interleaved stereo samples (in the present
 * implementation only copying to the headphone output, that is, to slots 0 and 2, is available).
 *
 * For both input and output transfers, audio double-buffering is simply implemented by using
 * a large (receive or transmit) buffer of size AUDIO_DMA_BUF_SIZE
 * and implementing half-buffer and full-buffer DMA callbacks:
 * - HAL_SAI_RxHalfCpltCallback() gets called whenever the receive (=input) buffer is half-full
 * - HAL_SAI_RxCpltCallback() gets called whenever the receive (=input) buffer is full
 *
 * As a result, one audio frame has a size of AUDIO_DMA_BUF_SIZE/2. But since one audio frame
 * contains interleaved L and R stereo samples, its true duration is AUDIO_DMA_BUF_SIZE/4.
 *
 * Example:
 * 		AUDIO_BUF_SIZE = 512 (=size of a stereo audio frame)
 * 		AUDIO_DMA_BUF_SIZE = 1024 (=size of the whole DMA buffer)
 * 		The duration of ONE audio frame is given by AUDIO_BUF_SIZE/2 = 256 samples, that is, 5.3ms at 48kHz.
 *
 * === interprocess communication ===
 *
 *  Communication b/w DMA IRQ Handlers and the main audio loop is carried out
 *  using the "audio_rec_buffer_state" global variable (using the input buffer instead of the output
 *  buffer is a matter of pure convenience, as both are filled at the same pace anyway).
 *
 *  This variable can take on three possible values:
 *  - BUFFER_OFFSET_NONE: initial buffer state at start-up, or buffer has just been transferred to/from DMA
 *  - BUFFER_OFFSET_HALF: first-half of the DMA buffer has just been filled
 *  - BUFFER_OFFSET_FULL: second-half of the DMA buffer has just been filled
 *
 *  The variable is written by HAL_SAI_RxHalfCpltCallback() and HAL_SAI_RxCpltCallback() audio in DMA transfer callbacks.
 *  It is read inside the main audio loop (see audioLoop()).
 *
 *  If RTOS is to used, Signals may be used to communicate between the DMA IRQ Handler and the main audio loop audioloop().
 *
 */

#include <audio.h>
#include <ui.h>
#include <stdio.h>
#include "string.h"
#include "math.h"
#include "bsp/disco_sai.h"
#include "bsp/disco_base.h"
#include "cmsis_os.h"
#include "arm_math.h"

extern SAI_HandleTypeDef hsai_BlockA2; // see main.c
extern SAI_HandleTypeDef hsai_BlockB2;
extern DMA_HandleTypeDef hdma_sai2_a;
extern DMA_HandleTypeDef hdma_sai2_b;

extern osThreadId defaultTaskHandle;
extern osThreadId uiTaskHandle;

// ---------- communication b/w DMA IRQ Handlers and the audio loop -------------

typedef enum {
	BUFFER_OFFSET_NONE = 0, BUFFER_OFFSET_HALF = 1, BUFFER_OFFSET_FULL = 2,
} BUFFER_StateTypeDef;
uint16_t audio_rec_buffer_state;




// ---------- DMA buffers ------------

// whole sample count in an audio frame: (beware: as they are interleaved stereo samples, true audio frame duration is given by AUDIO_BUF_SIZE/2)
#define AUDIO_BUF_SIZE   ((uint16_t)1024)
/* size of a full DMA buffer made up of two half-buffers (aka double-buffering) */
#define AUDIO_DMA_BUF_SIZE   (2 * AUDIO_BUF_SIZE)

#define FFT_Length (AUDIO_BUF_SIZE/2)

#define FRAME_SIZE 512    // 60 ms at 16 kHz
#define HOP_SIZE 256

#define RING_BUFF_SIZE 2048
// Constants for frame analysis and resynthesis
//#define FRAME_SIZE 960    // 60 ms at 16 kHz
//#define HOP_SIZE 320      // 20 ms at 16 kHz

// DMA buffers are in embedded RAM:
int16_t buf_input[AUDIO_DMA_BUF_SIZE];
int16_t buf_output[AUDIO_DMA_BUF_SIZE];
int16_t *buf_input_half = buf_input + AUDIO_DMA_BUF_SIZE / 2;
int16_t *buf_output_half = buf_output + AUDIO_DMA_BUF_SIZE / 2;

// Définition de la structure pour le calcul de la FFT
arm_rfft_fast_instance_f32 FFT_struct;
float32_t aFFT_Output_f32[FFT_Length];
float32_t aFFT_Input_f32[FFT_Length];

float32_t Voc_Input_f32[FRAME_SIZE];
float32_t Voc_Output_f32[FRAME_SIZE];
float32_t Voc_Output_BUFF_f32[2 * FRAME_SIZE];
float32_t Voc_Input_BUFF_f32[2 * FRAME_SIZE];


// ------------- scratch float buffer for long delays, reverbs or long impulse response FIR based on float implementations ---------

uint16_t scratch_offset = 0; // see doc in processAudio()
#define AUDIO_SCRATCH_SIZE   AUDIO_SCRATCH_MAXSZ_WORDS



// ------------ Private Function Prototypes ------------

static void noEffect(int16_t*, int16_t*);
static void simpleDelay(int16_t*, int16_t*);
static void accumulateInputLevels();
static float readFloatFromSDRAM(int pos);
static void writeFloatToSDRAM(float val, int pos);
static void processAudioWithPitchShift(int16_t *out, int16_t *in);
static void processModifiedAudioWithPitchShift(int16_t *out, int16_t *in);
static void vocoderLinBuff(int16_t *out, int16_t *in);
static void processModified3AudioWithPitchShift(int16_t *out, int16_t *in);
static void vocoderCircBuff(int16_t *out, int16_t *in);
static void vocoderCirc2Buff(int16_t *out, int16_t *in);

// ----------- Local vars ------------

static int count = 0; // debug
static double inputLevelL = 0;
static double inputLevelR = 0;
double inputLevelL2 = 0;
double inputLevelR2 = 0;

// ----------- Functions ------------

/**
 * This is the main audio loop (aka infinite while loop) which is responsible for real time audio processing tasks:
 * - transferring recorded audio from the DMA buffer to buf_input[]
 * - processing audio samples and writing them to buf_output[]
 * - transferring processed samples back to the DMA buffer
 */
void audioLoop() {

	uiDisplayBasic();

	/* Initialize SDRAM buffers */
	memset((int16_t*) AUDIO_SCRATCH_ADDR, 0, AUDIO_SCRATCH_SIZE * 2); // note that the size argument here always refers to bytes whatever the data type

	//audio_rec_buffer_state = BUFFER_OFFSET_NONE;


	// start SAI (audio) DMA transfers:
	startAudioDMA(buf_output, buf_input, AUDIO_DMA_BUF_SIZE);

	/* main audio loop */
	while (1) {

		/* calculate average input level over 20 audio frames */
		accumulateInputLevels();
		count++;
		if (count >= 20) {
			count = 0;
			inputLevelL *= 0.05;
			inputLevelR *= 0.05;
			uiDisplayInputLevel(inputLevelL, inputLevelR);
			inputLevelL = 0.;
			inputLevelR = 0.;
		}

		/* Wait until first half block has been recorded */
		while (audio_rec_buffer_state != BUFFER_OFFSET_HALF) {
			asm("NOP");
		}
		audio_rec_buffer_state = BUFFER_OFFSET_NONE;
		/* Copy recorded 1st half block */
		processAudio(buf_output, buf_input);

		/* Wait until second half block has been recorded */
		while (audio_rec_buffer_state != BUFFER_OFFSET_FULL) {
			asm("NOP");
		}
		audio_rec_buffer_state = BUFFER_OFFSET_NONE;
		/* Copy recorded 2nd half block */
		processAudio(buf_output_half, buf_input_half);

	}
}
//function to use the RTOS
void audioLoop2() {

	//uiDisplayBasic();



	/* Initialize SDRAM buffers */
	memset((int16_t*) AUDIO_SCRATCH_ADDR, 0, AUDIO_SCRATCH_SIZE * 2); // note that the size argument here always refers to bytes whatever the data type

	//audio_rec_buffer_state = BUFFER_OFFSET_NONE;

	// start SAI (audio) DMA transfers:
	startAudioDMA(buf_output, buf_input, AUDIO_DMA_BUF_SIZE);

	/* main audio loop */
	while (1) {

		/* calculate average input level over 20 audio frames */
		accumulateInputLevels();
		count++;
		if (count >= 20) {
			//osSignalSet(uiTaskHandle, 0x0003);
			count = 0;
			inputLevelL *= 0.05;
			inputLevelR *= 0.05;
			//uiDisplayInputLevel(inputLevelL, inputLevelR);
			inputLevelL2 = inputLevelL;
			inputLevelR2 = inputLevelR;
			inputLevelL = 0.;
			inputLevelR = 0.;
		}

		/* Wait until first half block has been recorded */
		/*while (audio_rec_buffer_state != BUFFER_OFFSET_HALF) {
			asm("NOP");
		}*/
		osSignalWait (0x0001, osWaitForever);
		//audio_rec_buffer_state = BUFFER_OFFSET_NONE;
		/* Copy recorded 1st half block */
		noEffect(buf_output, buf_input);
		//vocoderLinBuff(buf_output, buf_input);
		calculateFFT(buf_output);

		/* Wait until second half block has been recorded */
		/*while (audio_rec_buffer_state != BUFFER_OFFSET_FULL) {
			asm("NOP");
		}*/
		osSignalWait (0x0002, osWaitForever);
		//audio_rec_buffer_state = BUFFER_OFFSET_NONE;
		/* Copy recorded 2nd half block */
		noEffect(buf_output_half, buf_input_half);
		//vocoderLinBuff(buf_output_half, buf_input_half);
		calculateFFT(buf_output_half);

		}
}

/*
 * Fonction qui fait la TF
 */
void calculateFFT(int16_t *in){
	arm_rfft_fast_init_f32(&FFT_struct, FFT_Length);
	 for (int i = 0; i < FFT_Length; i++){
		 aFFT_Input_f32[i] = in[i];
	 }

	 arm_rfft_fast_f32(&FFT_struct, aFFT_Input_f32, aFFT_Output_f32, 0); //0 permet d'avoir la fft et 1 donne la fft inverse
	 arm_cmplx_mag_f32(aFFT_Output_f32, aFFT_Input_f32, FFT_Length/2);
	 osSignalSet(uiTaskHandle, 0x0003);
 }


/*
 * Update input levels from the last audio frame (see global variable inputLevelL and inputLevelR).
 * Reminder: audio samples are actually interleaved L/R samples,
 * with left channel samples at even positions,
 * and right channel samples at odd positions.
 */
static void accumulateInputLevels() {

	// Left channel:
	uint16_t lvl = 0;
	for (int i = 0; i < AUDIO_DMA_BUF_SIZE; i += 2) {
		int16_t v = (int16_t) buf_input[i];
		if (v > 0)
			lvl += v;
		else
			lvl -= v;
	}
	inputLevelL += (double) lvl / AUDIO_DMA_BUF_SIZE / (1 << 15);

	// Right channel:
	lvl = 0;
	for (int i = 1; i < AUDIO_DMA_BUF_SIZE; i += 2) {
		int16_t v = (int16_t) buf_input[i];
		if (v > 0)
			lvl += v;
		else
			lvl -= v;
	}
	inputLevelR += (double) lvl / AUDIO_DMA_BUF_SIZE / (1 << 15);
	;
}

// --------------------------- Callbacks implementation ---------------------------

/**
 * Audio IN DMA Transfer complete interrupt.
 */
void HAL_SAI_RxCpltCallback(SAI_HandleTypeDef *hsai) {
	//audio_rec_buffer_state = BUFFER_OFFSET_FULL;
	osSignalSet(defaultTaskHandle, 0x0002);
	return;
}

/**
 * Audio IN DMA Half Transfer complete interrupt.
 */
void HAL_SAI_RxHalfCpltCallback(SAI_HandleTypeDef *hsai) {
	//audio_rec_buffer_state = BUFFER_OFFSET_HALF;
	osSignalSet(defaultTaskHandle, 0x0001);
	return;
}

/* --------------------------- Audio "scratch" buffer in SDRAM ---------------------------
 *
 * The following functions allows you to use the external SDRAM as a "scratch" buffer.
 * There are around 7Mbytes of RAM available (~ 1' of stereo sound) which makes it possible to store signals
 * (either input or processed) over long periods of time for e.g. FIR filtering or long tail reverb's.
 */

/**
 * Read a 32 bit float from SDRAM at position "pos"
 */
static float readFloatFromSDRAM(int pos) {

	__IO float *pSdramAddress = (float*) AUDIO_SCRATCH_ADDR; // __IO is used to specify access to peripheral variables
	pSdramAddress += pos;
	//return *(__IO float*) pSdramAddress;
	return *pSdramAddress;

}

/**
 * Write the given 32 bit float to SDRAM at position "pos"
 */
static void writeFloatToSDRAM(float val, int pos) {

	__IO float *pSdramAddress = (float*) AUDIO_SCRATCH_ADDR;
	pSdramAddress += pos;/* USER CODE BEGIN Header_StartDefaultTask */

	//*(__IO float*) pSdramAddress = val;
	*pSdramAddress = val;


}

/**
 * Read a 16 bit integer from SDRAM at position "pos"
 */
static int16_t readint16FromSDRAM(int pos) {

	__IO int16_t *pSdramAddress = (int16_t*) AUDIO_SCRATCH_ADDR;
	pSdramAddress += pos;
	//return *(__IO int16_t*) pSdramAddress;
	return *pSdramAddress;

}

/**
 * Write the given 16 bit integer to the SDRAM at position "pos"
 */
static void writeint16ToSDRAM(int16_t val, int pos) {

	__IO int16_t *pSdramAddress = (int16_t*) AUDIO_SCRATCH_ADDR;
	pSdramAddress += pos;
	//*(__IO int16_t*) pSdramAddress = val;
	*pSdramAddress = val;

}

// --------------------------- AUDIO ALGORITHMS ---------------------------

/**
 * This function is called every time an audio frame
 * has been filled by the DMA, that is,  AUDIO_BUF_SIZE samples
 * have just been transferred from the CODEC
 * (keep in mind that this number represents interleaved L and R samples,
 * hence the true corresponding duration of this audio frame is AUDIO_BUF_SIZE/2 divided by the sampling frequency).
 */

static void noEffect(int16_t *out, int16_t *in) {
	LED_On(); // for oscilloscope measurements...
	for (int n = 0; n < AUDIO_BUF_SIZE; n++) {
		out[n] = in[n];
	}
	LED_Off();
}

// fonction delay
int i =0;
static void simpleDelay(int16_t *out, int16_t *in) {

	int Delay = 0.5 * 16000;

	LED_On(); // for oscilloscope measurements...
	for (int n = 0; n < AUDIO_BUF_SIZE; n++){
		writeFloatToSDRAM(in[n], (i+Delay)%100000);
		out[n] = in[n] + 0.5* readFloatFromSDRAM(i);
		i += 1;
		i = i % 100000;
	}
	LED_Off();
}



static void processAudioWithPitchShift(int16_t *out, int16_t *in) {
    float pitchShiftFactor = 0.5;  // Modifier le facteur de pitch shift selon vos besoins
    int circBufferSize = 16384; //
    memset(Voc_Output_BUFF_f32, 0, sizeof(Voc_Output_BUFF_f32));

    int i = 0;

    // Initialiser les instances FFT
    arm_rfft_fast_instance_f32 FFT_struct;
    arm_rfft_fast_init_f32(&FFT_struct, FRAME_SIZE);

    LED_On(); // for oscilloscope measurements...


	for (int n = 0; n < FRAME_SIZE; n++) {

		// Appliquez la fenêtre de Hanning
		float hanningWindow = 0.5 * (1.0 - cos(2.0 * M_PI * n / (float)(FRAME_SIZE - 1)));
		Voc_Input_f32[n]= hanningWindow * in[n];
	}
	arm_rfft_fast_f32(&FFT_struct, Voc_Input_f32, Voc_Output_f32, 0);
	//On peut maintenant traiter le signal sur Voc_Output32
	/*for (int n = 0; n < FRAME_SIZE/ 2 + 1; n++) {
		int newBin = (int)(n * pitchShiftFactor) % (FRAME_SIZE / 2 + 1);
		Voc_Input_f32[newBin] = Voc_Output_f32[n];
	}*/

	arm_rfft_fast_f32(&FFT_struct, Voc_Output_f32, Voc_Input_f32, 1);

	for (int n = 0; n < FRAME_SIZE; n++) {
		int reader = readFloatFromSDRAM(i+n);
		if (reader ==0) {
				writeFloatToSDRAM(Voc_Input_f32[n],i+n);
		}
		else {
			writeFloatToSDRAM(reader+Voc_Input_f32[n],i+n);
		}

	}
	i = (i + HOP_SIZE) % circBufferSize;
	for (int n = 0; n < AUDIO_BUF_SIZE; n++){
	    	out[n]=readFloatFromSDRAM(n);
    }
    i = (i + HOP_SIZE) % circBufferSize;


    LED_Off();
}


static void processModifiedAudioWithPitchShift(int16_t *out, int16_t *in) {
    float pitchShiftFactor = 0.5;  // Modifier le facteur de pitch shift selon vos besoins
    //int circBufferSize = 16384; //
    memset(Voc_Output_BUFF_f32, 0, sizeof(Voc_Output_BUFF_f32));

    // Initialiser les instances FFT
    arm_rfft_fast_instance_f32 FFT_struct;
    arm_rfft_fast_init_f32(&FFT_struct, FRAME_SIZE);

    LED_On(); // for oscilloscope measurements...
    //crée mon buffer d'entrée
    for (int i = 0; i < 2*AUDIO_BUF_SIZE; i++){
    	Voc_Input_BUFF_f32[i]=in[i];
    }
    for (int j = 0; j < AUDIO_BUF_SIZE; j+=HOP_SIZE){
		for (int n = 0; n < FRAME_SIZE; n++) {
			// On applique la fenêtre de Hanning sur chaque frame
			float hanningWindow = 0.5 * (1.0 - cos(2.0 * M_PI * n / (float)(FRAME_SIZE - 1)));
			Voc_Input_f32[n]= hanningWindow * Voc_Input_BUFF_f32[j+n];
		}
		//On fait une fft pour pouvoir traiter le signal en fréquence sur chaque frame
		arm_rfft_fast_f32(&FFT_struct, Voc_Input_f32, Voc_Output_f32, 0);

		//On peut maintenant traiter le signal sur Voc_Output32
		/*for (int n = 0; n < FRAME_SIZE/ 2 + 1; n++) {
			int newBin = (int)(n * pitchShiftFactor) % (FRAME_SIZE / 2 + 1);
			Voc_Input_f32[newBin] = Voc_Output_f32[n];
		}*/

		//On fait une fft inverse pour repasser en temporel
		arm_rfft_fast_f32(&FFT_struct, Voc_Output_f32, Voc_Input_f32, 1);

		//On place le signal correctement dans le buffer de sortie tout en prenant compte l'overlapping
		for (int n = 0; n < FRAME_SIZE; n++) {
			int reader = Voc_Output_BUFF_f32[j+n];
			if (reader ==0) {
				Voc_Output_BUFF_f32[j+n] = Voc_Input_f32[n];
			}
			else {
				Voc_Output_BUFF_f32[j+n] = Voc_Input_f32[n]+Voc_Output_BUFF_f32[j+n];
			}

		}
    }
    //Quand le buffer de sortie est remplie correctement, on lis depuis ce derrnier
	for (int n = 0; n < AUDIO_BUF_SIZE; n++){
		out[n]=Voc_Output_BUFF_f32[n];
	}

    LED_Off();
}

//vocoder without circular buffers
static void vocoderLinBuff(int16_t *out, int16_t *in) {
    float pitchShiftFactor =2;  // Le facteur de pitch shift peut être modifié: 1=même signal, 2=augmentation d'une octave, 0.5=diminution d'une octave.

    //initialise le buffer de sortie
    memset(Voc_Output_BUFF_f32, 0, sizeof(Voc_Output_BUFF_f32));

    // Initialiser les instances FFT
    arm_rfft_fast_init_f32(&FFT_struct, FRAME_SIZE);

    LED_On(); // for oscilloscope measurements...
    //crée le buffer d'entrée
    for (int n = 0; n <AUDIO_BUF_SIZE; n++){
    	writeFloatToSDRAM(in[n],n);

    }
    for (int j = 0; j < AUDIO_BUF_SIZE; j+=HOP_SIZE){
		for (int n = 0; n < FRAME_SIZE; n++) {
			// On applique la fenêtre de Hanning sur chaque frame
			float hanningWindow = 0.5 * (1.0 - cos(2.0 * M_PI * n / (float)(FRAME_SIZE - 1)));
			Voc_Input_f32[n]= hanningWindow * readFloatFromSDRAM(j+n);
		}
		//On fait une fft pour pouvoir traiter le signal en fréquence sur chaque frame
		arm_rfft_fast_f32(&FFT_struct, Voc_Input_f32, Voc_Output_f32, 0);

		//On peut maintenant traiter le signal sur Voc_Output32. Cette version du vocoder multiplie simplement la fréquence.
		for (int n = 0; n < FRAME_SIZE/ 2 + 1; n++) {
			int newBin = (int)(n * pitchShiftFactor) % (FRAME_SIZE / 2 + 1);
			Voc_Input_f32[newBin] = Voc_Output_f32[n];
		}

		//On fait une fft inverse pour repasser en temporel
		arm_rfft_fast_f32(&FFT_struct, Voc_Input_f32, Voc_Output_f32, 1);

		//On place le signal correctement dans le buffer de sortie tout en prenant compte l'overlapping
		for (int n = 0; n < FRAME_SIZE; n++) {
			int reader = Voc_Output_BUFF_f32[j+n];

			if (reader ==0) {
				Voc_Output_BUFF_f32[j+n] = Voc_Output_f32[n];
			}
			else {
				Voc_Output_BUFF_f32[j+n] = Voc_Output_f32[n]+Voc_Output_BUFF_f32[j+n]; //on fait de l'overlapping add
			}
		}
    }
    //Quand le buffer de sortie est remplie correctement, on lis depuis ce derrnier
	for (int n = 0; n < AUDIO_BUF_SIZE; n++){
		out[n]=Voc_Output_BUFF_f32[n];
	}

    LED_Off();
}



int l = 5000; // indice d'écriture du buffer d'entrée
int p = 0;    // indice d'écriture du buffer de sortie
int p_ant = 0;
int r = 0;    // indice du moment de lecture du buffer circulaire de sortie

static void vocoderCircBuff(int16_t *out, int16_t *in) {
    float pitchShiftFactor = 1;  // Modifier le facteur de pitch shift selon vos besoins
    // int circBufferSize = 16384; //
    // memset(Voc_Output_BUFF_f32, 0, sizeof(Voc_Output_BUFF_f32));
    // memset((int16_t*) AUDIO_SCRATCH_ADDR, 0, AUDIO_SCRATCH_SIZE * 2);
    // Initialiser les instances FFT
    arm_rfft_fast_init_f32(&FFT_struct, FRAME_SIZE);



    // crée mon buffer d'entrée
    for (int n = 0; n < AUDIO_BUF_SIZE; n++) {
        writeFloatToSDRAM(in[n], l);
        if (l < 5000+RING_BUFF_SIZE) {
            l += 1;
        } else {
            l = 5000;
        }
    }
    while ( p < p_ant + AUDIO_BUF_SIZE ){

    	// for (int j = 0; j < AUDIO_BUF_SIZE; j+=HOP_SIZE){
		for (int n = 0; n < FRAME_SIZE; n++) {
			// On applique la fenêtre de Hanning sur chaque frame
			float hanningWindow = 0.5 * (1.0 - cos(2.0 * M_PI * n / (float)(FRAME_SIZE - 1)));
			Voc_Input_f32[n] = hanningWindow * readFloatFromSDRAM(((p + n)%RING_BUFF_SIZE)+5000);
		}

		// On fait une fft pour pouvoir traiter le signal en fréquence sur chaque frame
		arm_rfft_fast_f32(&FFT_struct, Voc_Input_f32, Voc_Output_f32, 0);

		// On peut maintenant traiter le signal sur Voc_Output32

		/*for (int n = 0; n < FRAME_SIZE / 2 + 1; n++) {
			int newBin = (int)(n * pitchShiftFactor) % (FRAME_SIZE / 2 + 1);
			Voc_Input_f32[newBin] = Voc_Output_f32[n];
		}*/

		// On fait une fft inverse pour repasser en temporel
		arm_rfft_fast_f32(&FFT_struct, Voc_Output_f32, Voc_Input_f32, 1);

		// On place le signal correctement dans le buffer de sortie tout en prenant compte l'overlapping
		for (int n = 0; n < FRAME_SIZE; n++) {
			// int reader = Voc_Output_BUFF_f32[j+n];
			int reader = readFloatFromSDRAM((p + n)%RING_BUFF_SIZE);
			if (reader == 0) {
				writeFloatToSDRAM(Voc_Input_f32[n], (p + n)%RING_BUFF_SIZE);
				// Voc_Output_BUFF_f32[j+n] = Voc_Output_f32[n];
			} else {
				int overlap = readFloatFromSDRAM((p + n)%RING_BUFF_SIZE);
				writeFloatToSDRAM(Voc_Input_f32[n] + overlap, (p + n)%RING_BUFF_SIZE);
				// Voc_Output_BUFF_f32[j+n] = Voc_Output_f32[n]+Voc_Output_BUFF_f32[j+n];
			}
		}

		if (p < RING_BUFF_SIZE) {
			p += HOP_SIZE;
		}
		else {
			p = 0;
		}
		//
    }
    r=p % (AUDIO_BUF_SIZE);
    // }
    LED_On();
    p_ant = p;
    // Quand le buffer de sortie est rempli correctement, on lit depuis ce dernier

    if (r == 0 && p !=0) {

        for (int n = 0; n < AUDIO_BUF_SIZE; n++) {
            out[n] = readFloatFromSDRAM(p-AUDIO_BUF_SIZE+n);
            writeFloatToSDRAM(0, p-AUDIO_BUF_SIZE+n);

             }
    }
    if (r == 0 && p ==0) {
        //LED_On(); // for oscilloscope measurements...
    	for (int n = 0; n < AUDIO_BUF_SIZE; n++) {
    	            out[n] = readFloatFromSDRAM(RING_BUFF_SIZE-AUDIO_BUF_SIZE+n);
    	            writeFloatToSDRAM(0, RING_BUFF_SIZE-AUDIO_BUF_SIZE+n);
//    	            LED_Off();
    	             }
    }


}

#define MAX_BUFFER_SIZE     1024  // Choisir la taille maximale du buffer circulaire

/*static float Voc_Input_f32[FRAME_SIZE];
static float Voc_Output_f32[FRAME_SIZE];
static float Voc_Output_BUFF_f32[MAX_BUFFER_SIZE];*/

// Indices pour le buffer circulaire
static int writeIndex = 0;
static int readIndex = 0;

static void vocoderCirc2Buff(int16_t *out, int16_t *in) {
    float pitchShiftFactor = 1;

    // Initialise le buffer de sortie
    memset(Voc_Output_BUFF_f32, 0, sizeof(Voc_Output_BUFF_f32));

    // Initialise les instances FFT
    arm_rfft_fast_init_f32(&FFT_struct, FRAME_SIZE);

    LED_On();

    for (int n = 0; n < AUDIO_BUF_SIZE; n++) {
        writeFloatToSDRAM(in[n], n);
    }

    while (readIndex < AUDIO_BUF_SIZE) {
        // Crée le buffer d'entrée
        for (int n = 0; n < FRAME_SIZE; n++) {
            float hanningWindow = 0.5 * (1.0 - cos(2.0 * M_PI * n / (float)(FRAME_SIZE - 1)));
            Voc_Input_f32[n] = hanningWindow * readFloatFromSDRAM(readIndex + n);
        }

        // FFT pour traiter le signal en fréquence sur chaque frame
        arm_rfft_fast_f32(&FFT_struct, Voc_Input_f32, Voc_Output_f32, 0);

        // Traitement du signal sur Voc_Output32 (multiplication de la fréquence)
        for (int n = 0; n < FRAME_SIZE / 2 + 1; n++) {
            int newBin = (int)(n * pitchShiftFactor) % (FRAME_SIZE / 2 + 1);
            Voc_Input_f32[newBin] = Voc_Output_f32[n];
        }

        // FFT inverse pour repasser en temporel
        arm_rfft_fast_f32(&FFT_struct, Voc_Input_f32, Voc_Output_f32, 1);

        // Placement du signal dans le buffer de sortie en prenant en compte l'overlapping
        for (int n = 0; n < FRAME_SIZE; n++) {
            Voc_Output_BUFF_f32[writeIndex] += Voc_Output_f32[n];
            writeIndex = (writeIndex + 1) % MAX_BUFFER_SIZE;
        }

        readIndex += HOP_SIZE;
    }

    // Lecture depuis le buffer circulaire pour remplir le buffer de sortie
    for (int n = 0; n < AUDIO_BUF_SIZE; n++) {
        out[n] = Voc_Output_BUFF_f32[readIndex];
        readIndex = (readIndex + 1) % MAX_BUFFER_SIZE;
    }

    LED_Off();
}
