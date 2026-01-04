#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <sys/timeb.h>
using namespace std;

#define MAX_ITERATIONS ULLONG_MAX // numarul maxim de iteratii poate fi cel mult 2^(64)-1
#define SHA_SIZE 20 // SHA1 afiseaza un digest de 20 bytes, sau 20 octeti

#define ZEROS_TO_FIND 4
#define BUFFER "Tudor"
#define BUFFER_SIZE 12

typedef unsigned char BYTE;
typedef unsigned int  WORD;
typedef unsigned long long LONG;

// Implementarea SHA1
typedef struct {
	BYTE data[64];
	WORD data_length;
	LONG bit_length;
	WORD state[5];
	WORD round_constant[4];
} CUDA_SHA1_CONTEXT;

#ifndef ROTATE_LEFT
#define ROTATE_LEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

__device__  __forceinline__ void cuda_sha1_transform(CUDA_SHA1_CONTEXT* context, const BYTE data[])
{
	WORD state0, state1, state2, state3, state4, i, j, transform, message[80];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		message[i] = (data[j] << 24) + (data[j + 1] << 16) + (data[j + 2] << 8) + (data[j + 3]);
	for (; i < 80; ++i) {
		message[i] = (message[i - 3] ^ message[i - 8] ^ message[i - 14] ^ message[i - 16]);
		message[i] = (message[i] << 1) | (message[i] >> 31);
	}

	state0 = context->state[0];
	state1 = context->state[1];
	state2 = context->state[2];
	state3 = context->state[3];
	state4 = context->state[4];

	for (i = 0; i < 20; ++i) {
		transform = ROTATE_LEFT(state0, 5) + ((state1 & state2) ^ (~state1 & state3)) + state4 + context->round_constant[0] + message[i];
		state4 = state3;
		state3 = state2;
		state2 = ROTATE_LEFT(state1, 30);
		state1 = state0;
		state0 = transform;
	}
	for (; i < 40; ++i) {
		transform = ROTATE_LEFT(state0, 5) + (state1 ^ state2 ^ state3) + state4 + context->round_constant[1] + message[i];
		state4 = state3;
		state3 = state2;
		state2 = ROTATE_LEFT(state1, 30);
		state1 = state0;
		state0 = transform;
	}
	for (; i < 60; ++i) {
		transform = ROTATE_LEFT(state0, 5) + ((state1 & state2) ^ (state1 & state3) ^ (state2 & state3)) + state4 + context->round_constant[2] + message[i];
		state4 = state3;
		state3 = state2;
		state2 = ROTATE_LEFT(state1, 30);
		state1 = state0;
		state0 = transform;
	}
	for (; i < 80; ++i) {
		transform = ROTATE_LEFT(state0, 5) + (state1 ^ state2 ^ state3) + state4 + context->round_constant[3] + message[i];
		state4 = state3;
		state3 = state2;
		state2 = ROTATE_LEFT(state1, 30);
		state1 = state0;
		state0 = transform;
	}

	context->state[0] += state0;
	context->state[1] += state1;
	context->state[2] += state2;
	context->state[3] += state3;
	context->state[4] += state4;
}

__device__ void cuda_sha1_initialization(CUDA_SHA1_CONTEXT* context)
{
	context->data_length = 0;
	context->bit_length = 0;
	context->state[0] = 0x67452301;
	context->state[1] = 0xEFCDAB89;
	context->state[2] = 0x98BADCFE;
	context->state[3] = 0x10325476;
	context->state[4] = 0xc3d2e1f0;
	context->round_constant[0] = 0x5a827999;
	context->round_constant[1] = 0x6ed9eba1;
	context->round_constant[2] = 0x8f1bbcdc;
	context->round_constant[3] = 0xca62c1d6;
}

__device__ void cuda_sha1_update(CUDA_SHA1_CONTEXT* context, const BYTE data[], size_t length)
{
	for (size_t i = 0; i < length; ++i) {
		context->data[context->data_length] = data[i];
		context->data_length++;
		if (context->data_length == 64) {
			cuda_sha1_transform(context, context->data);
			context->bit_length += 512;
			context->data_length = 0;
		}
	}
}

__device__ void cuda_sha1_final(CUDA_SHA1_CONTEXT* context, BYTE hash[])
{
	WORD i;

	i = context->data_length;

	// Adauga orice date, care au mai ramas in buffer.
	if (context->data_length < 56) {
		context->data[i++] = 0x80;
		while (i < 56)
			context->data[i++] = 0x00;
	}
	else {
		context->data[i++] = 0x80;
		while (i < 64)
			context->data[i++] = 0x00;
		cuda_sha1_transform(context, context->data);
		memset(context->data, 0, 56);
	}

	// Adauga la padding lungimea mesajului in biti si transforma.
	context->bit_length += context->data_length * 8;
	context->data[63] = context->bit_length;
	context->data[62] = context->bit_length >> 8;
	context->data[61] = context->bit_length >> 16;
	context->data[60] = context->bit_length >> 24;
	context->data[59] = context->bit_length >> 32;
	context->data[58] = context->bit_length >> 40;
	context->data[57] = context->bit_length >> 48;
	context->data[56] = context->bit_length >> 56;
	cuda_sha1_transform(context, context->data);

	// Inverseaza toti octetii la copierea starii finale in hash-ul de iesire.
	for (i = 0; i < 4; ++i) {
		hash[i] = (context->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4] = (context->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8] = (context->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (context->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (context->state[4] >> (24 - i * 8)) & 0x000000ff;
	}
}

__device__ void data_reverse(BYTE string[], int size)
{
	int start = 0;
	int end = size - 1;
	while (start < end) {
		char h = *(string + start), t = *(string + end);
		*(string + start) = t;
		*(string + end) = h;
		start++;
		end--;
	}
}

__device__ BYTE* data_int_to_byte(size_t number, BYTE* string, int* size)
{
	int i = 0;

	// Gestioneaza 0 in mod explicit. In caz contrar, pentru 0 se va afisa un sir gol.
	if (number == 0) {
		string[i++] = '0';
		return string;
	}

	// Proceseaza cifrele in mod individual
	while (number != 0) {
		int remainder = number % 10;
		string[i++] = (remainder > 9) ? (remainder - 10) + 'a' : remainder + '0';
		number = number / 10;
	}

	// Intoarce sirul
	data_reverse(string, i);
	*size = i;

	return string;
}


// Functie kernel pentru gasirea unui nonce, care genereaza un hash SHA-1, terminat cu un numar specific de zerouri
__global__ void find_nonce(size_t* result, BYTE* hash, bool* found, size_t warp) {
	CUDA_SHA1_CONTEXT context;            // Contextul SHA-1 pentru efectuarea operatiilor hash
	BYTE checksum[SHA_SIZE];      // Buffer-ul, care stocheaza hash-ul calculat

	// Calculeaza un ID unic de thread pentru thread-ul CUDA curent
	unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculeaza nonce-ul pe baza ID-ului thread-ului and warp-ului, warp-ul fiind o unitate fundamentala de executie pe un GPU, care consta din 32 thread-uri, care executa aceeasi instructiune simultan pe date diferite
	size_t nonce_source = thread + warp;

	// Se pregateste input-ul pentru hashing
	int nonce_size = 0;
	BYTE nonce[SHA_SIZE];
	BYTE input[BUFFER_SIZE + SHA_SIZE]; // Se combina datele buffer-ului si nonce-ul

	data_int_to_byte(nonce_source, nonce, &nonce_size);  // Converteste nonce-ul intreg intr-un vector
	memcpy(input, (BYTE*)BUFFER, BUFFER_SIZE); // Copiaza buffer-ul initial in input
	memcpy(input + BUFFER_SIZE, nonce, nonce_size); // Adauga nonce-ul la input

	// Initializeaza buffer-ul hash-ului la zero
	memset(checksum, 0x0, SHA_SIZE);

	// Se efectueaza calculul hash-ului SHA1
	cuda_sha1_initialization(&context);
	cuda_sha1_update(&context, input, BUFFER_SIZE + nonce_size);
	cuda_sha1_final(&context, checksum);

	// Verifica daca hash-ul calculat se termina cu numarul necesar de zero-uri
	bool suffix_matches = true;
	for (int i = 0; i < ZEROS_TO_FIND/2+1; i++) {
		// Verifica ultimii octeti ZEROS_TO_FIND
		if (checksum[SHA_SIZE - i - 1] != 0 || checksum[SHA_SIZE- i-ZEROS_TO_FIND/2- ZEROS_TO_FIND%2] != 0) {
			suffix_matches = false; // Toti octetii trebuie sa fie exact zero
			break;
		}
	}

	// Daca a fost gasit un nonce valid, se actualizeaza rezultatul, care este criptata prin functia hash, si marcata ca fiind gasit
	if (suffix_matches) {
		*found = true;                // Solutia a fost gasita
		*result = nonce_source;       // Salveaza nonce-ul
		memcpy(hash, checksum, sizeof(checksum));     // Copiaza hash-ul in buffer-ul de output

	}
}

// Functie, care determina dimensiunile optime ale grid-ului si block-urilor pentru executia CUDA
void get_optimal_sizes(int* grid_size, int* block_size) {
	cudaDeviceProp deviceProperties; // Structura care pastreaza proprietatile device-ului

	// Proprietatile primului device CUDA
	if (cudaSuccess != cudaGetDeviceProperties(&deviceProperties, 0)) {
		// Setarea valorilor implicite pentru marimile grid-ului si block-ului
		*grid_size = 32;
		*block_size = 32;
		return;
	}

	// Calculati dimensiunea optima a block-ului și dimensiunea grid-ului pentru kernel-ul find_nonce
	cudaOccupancyMaxPotentialBlockSize(grid_size, block_size, find_nonce);
	*grid_size = 2048; // Marimea gridului pentru GPU
}

// Functia principala
int main(int argc, char** argv) {
	bool host_found = false;          // Host flag, care indica daca un nonce a fost gasit
	size_t host_nonce = 0;            // Variabila care stocheaza nonce-ul rezultat
	size_t nonce_size = sizeof(size_t); // Marimea nonce-ului in octeti
	int grid_size;                 // Dimensiunea grid-ului pentru execuția kernelului CUDA
	int block_size;                // Dimensiunea block-ului pentru execuția kernelului CUDA

	size_t warp = 0;             // Pasul curent
	size_t thread_count = 0;           // Numar total de thread-uri

	struct timeb start, end;       // Variabilele de masurare a timpului
	double seconds = 0;            // Timpul total scurs

	cudaError_t status = cudaSuccess; // Variabila starii CUDA

	// Calculeaza marimile optime ale grid-ului si block-ului
	get_optimal_sizes(&grid_size, &block_size);

	// Buffer-ul host pentru stocarea hash-ului rezultat
	BYTE* host_digest = (BYTE*)malloc(SHA_SIZE);
	memset(host_digest, 0, SHA_SIZE); // Initializeaza digest-ul la zero

	// Alocari de memorie in dispozitiv, pentru nonce, hash si flag-ul gasit
	size_t* device_nonce;
	bool* device_found;
	BYTE* device_digest;
	status = cudaMalloc((void**)&device_nonce, nonce_size);
	status = cudaMalloc((void**)&device_digest, SHA_SIZE);
	status = cudaMalloc((void**)&device_found, sizeof(bool));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc a esuat!"); // Afiseaza eroarea daca alocarea de memorie esueaza
		return -1;
	}

	// Initializeaza memoria dispozitivului pentru flag-ul gasit
	cudaMemcpy(device_found, &host_found, sizeof(bool), cudaMemcpyHostToDevice);

	// Se incepe cronometrarea
	ftime(&start);
	thread_count = grid_size * block_size; // Numar total de thread-uri

	// Lanseaza kernelul iterativ pana cand se gaseste o solutie sau se atinge numarul maxim de iteratii
	for (size_t i = 0; i <= MAX_ITERATIONS && !host_found; i++) {
		find_nonce << <grid_size, block_size >> > (device_nonce, device_digest, device_found, warp);
		cudaDeviceSynchronize(); // Se asteapta pana la finalizarea executiei kernel-ului

		// Se verifica erorile de lansare a kernel-ului
		cudaError_t kernel_err = cudaGetLastError();
		if (kernel_err != cudaSuccess) {
			printf("Eroare de kernel CUDA: %s\n", cudaGetErrorString(kernel_err));
			break;
		}

		// Copiaza flag-ul gasit inapoi pe host
		cudaMemcpy(&host_found, device_found, sizeof(bool), cudaMemcpyDeviceToHost);
		warp += thread_count; // Incrementeaza warp-ul pentru urmatoarea serie de nonce-uri
	}

	// Copiaza hash-ul si nonce-ul rezultate inapoi in CPU
	cudaMemcpy(host_digest, device_digest, SHA_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(&host_nonce, device_nonce, nonce_size, cudaMemcpyDeviceToHost);

	// Finalizeaza cronometrarea
	ftime(&end);
	seconds = end.time - start.time + ((double)end.millitm - (double)start.millitm) / 1000.0;

	// Afiseaza rezultatele si performanta
	printf(" Pentru buffer-ul %s hash-ul care se incheie in %d zero-uri s-a terminat calculand:\n \
un hashrate de %ld hashuri/secunda. Durata a fost de %.2f secunde\n \
Thread-uri cu marimea gridului de %d si marimea block-ului de %d\n",
		BUFFER,(int)ZEROS_TO_FIND,(long)(warp / seconds), seconds, grid_size, block_size);

	// Afiseaza detaliile, odata ce un nonce a fost gasit
	if (host_found) {
		char hex_result[SHA_SIZE * 2 + 1] = { 0 }; // Buffer-ul pentru reprezentarea in siruri si in hexazecimal a hash-ului
		for (int offset = 0; offset < SHA_SIZE; offset++) {
			sprintf((hex_result + (2 * offset)), "%02x", host_digest[offset] & 0xff); // Converteste in hexazecimal
		}
		printf(" Nonce: %zu. Digest: %s\n", host_nonce, hex_result); // Afiseaza nonce-ul si hash-ul
	}
	else {
		printf("Nu s-a gasit un nonce, astfel incat digest-ul SHA sa se incheie cu %d zero-uri\n", ZEROS_TO_FIND);
	}

	// Elibereaza memoria alocata.
	free(host_digest);
	cudaFree(device_nonce);
	cudaFree(device_digest);
	cudaFree(device_found);

	return status;
}
