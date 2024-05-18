#include <raylib.h>
#include "cuda_runtime.h"
#include <iostream>

const int BLUR_SIZE = 20;
const int CHANNELS = 4; // RGBA

__global__ void BlurImageKernel(unsigned char* in, unsigned char* out, int w, int h) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (col < w && row < h) {
        for (int c = 0; c < CHANNELS; ++c) {
            int pixVal = 0;
            int pixels = 0;
            for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                        pixVal += in[(curRow * w + curCol) * CHANNELS + c];
                        pixels++;
                    }
                }
            }
            out[(row * w + col) * CHANNELS + c] = pixVal / pixels;
        }
    }
}

int main() {
    // Load the image
    Image goku = LoadImage("C:/Users/ochoa/Desktop/Programing/Cuda/Learning/CudaColorToGrayScale/goku.png");

    if (goku.data == NULL) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }

    // Ensure the image is in the correct format
    ImageFormat(&goku, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);

    // Get image data
    unsigned char* imgData = (unsigned char*)goku.data;
    int imgWidth = goku.width;
    int imgHeight = goku.height;
    int imgSize = imgWidth * imgHeight * CHANNELS * sizeof(unsigned char);

    // Initialize the window to match the image size
    InitWindow(imgWidth, imgHeight, "Blurring Image");

    // Allocate memory on the device
    unsigned char* d_in, * d_out;
    cudaMalloc((void**)&d_in, imgSize);
    cudaMalloc((void**)&d_out, imgSize);

    // Copy image data to the device
    cudaMemcpy(d_in, imgData, imgSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((imgWidth + blockSize.x - 1) / blockSize.x, (imgHeight + blockSize.y - 1) / blockSize.y, 1);

    // Launch the kernel
    BlurImageKernel << <gridSize, blockSize >> > (d_in, d_out, imgWidth, imgHeight);

    // Allocate memory for the result on the host
    unsigned char* blurredImgData = new unsigned char[imgWidth * imgHeight * CHANNELS];

    // Copy the result back to the host
    cudaMemcpy(blurredImgData, d_out, imgSize, cudaMemcpyDeviceToHost);

    // Create a new image for the blurred result
    Image blurredImg = {
        blurredImgData,
        imgWidth,
        imgHeight,
        1,
        PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    };

    // Convert the image to a texture
    Texture2D texture = LoadTextureFromImage(blurredImg);

    // Unload the original image and device memory
    UnloadImage(goku);
    cudaFree(d_in);
    cudaFree(d_out);

    // Main game loop
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(WHITE);

        // Draw the texture
        DrawTexture(texture, 0, 0, WHITE);

        EndDrawing();
    }

    // Unload the texture
    UnloadTexture(texture);

    // Clean up the blurred image data
    delete[] blurredImgData;

    // Close the window and terminate Raylib
    CloseWindow();

    return 0;
}
