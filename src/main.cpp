#include <M5GFX.h>
#include <../CNN/mnist_float32.h>
#include <math.h>

M5GFX display;

#define GRID_SIZE 28

float grid[GRID_SIZE][GRID_SIZE];

void printGrid();
void runCNN();


// =====================================================
// SOFTMAX (stable)
// =====================================================
void softmax(float *input, float *output, int size) {
  float maxVal = input[0];
  for(int i = 1; i < size; i++){
    if(input[i] > maxVal) maxVal = input[i];
  }
  float sum = 0.0f;
  for(int i = 0; i < size; i++){
    output[i] = expf(input[i] - maxVal);
    sum += output[i];
  }
  for(int i = 0; i < size; i++){
    output[i] /= sum;
  }
}


// =====================================================
// GAUSSIAN BLUR 5x5
// Lisse le dessin pour coller aux images MNIST
// sigma ≈ 0.8, kernel normalisé
// =====================================================
void gaussianBlur(float src[GRID_SIZE][GRID_SIZE], float dst[GRID_SIZE][GRID_SIZE]) {

  // Kernel gaussien 5x5 (sigma ≈ 0.8), pré-calculé et normalisé
  static const float kernel[5][5] = {
    { 0.00296902f, 0.01330621f, 0.02193823f, 0.01330621f, 0.00296902f },
    { 0.01330621f, 0.05963430f, 0.09832033f, 0.05963430f, 0.01330621f },
    { 0.02193823f, 0.09832033f, 0.16210282f, 0.09832033f, 0.02193823f },
    { 0.01330621f, 0.05963430f, 0.09832033f, 0.05963430f, 0.01330621f },
    { 0.00296902f, 0.01330621f, 0.02193823f, 0.01330621f, 0.00296902f }
  };

  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){

      float acc   = 0.0f;
      float wsum  = 0.0f;

      for(int ky = -2; ky <= 2; ky++){
        for(int kx = -2; kx <= 2; kx++){
          int nx = x + kx;
          int ny = y + ky;
          if(nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE){
            float w = kernel[ky + 2][kx + 2];
            acc  += src[ny][nx] * w;
            wsum += w;
          }
        }
      }

      dst[y][x] = (wsum > 0.0f) ? (acc / wsum) : 0.0f;
    }
  }
}


// =====================================================
// BOUNDING BOX
// Retourne les coordonnées min/max du trait dessiné
// Retourne false si la grille est vide
// =====================================================
bool getBoundingBox(float src[GRID_SIZE][GRID_SIZE],
                    int &xmin, int &ymin, int &xmax, int &ymax) {

  xmin = GRID_SIZE; ymin = GRID_SIZE;
  xmax = -1;        ymax = -1;

  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      if(src[y][x] > 0.05f){        // seuil pour ignorer le bruit
        if(x < xmin) xmin = x;
        if(x > xmax) xmax = x;
        if(y < ymin) ymin = y;
        if(y > ymax) ymax = y;
      }
    }
  }

  return (xmax >= xmin && ymax >= ymin);
}


// =====================================================
// CENTRAGE + MISE À L'ÉCHELLE (style MNIST)
// Le chiffre est scalé pour tenir dans 20x20 pixels
// puis centré dans la grille 28x28 par son centroïde
// =====================================================
void centerAndScale(float src[GRID_SIZE][GRID_SIZE], float dst[GRID_SIZE][GRID_SIZE]) {

  // Grille de destination initialisée à 0
  for(int y = 0; y < GRID_SIZE; y++)
    for(int x = 0; x < GRID_SIZE; x++)
      dst[y][x] = 0.0f;

  int xmin, ymin, xmax, ymax;
  if(!getBoundingBox(src, xmin, ymin, xmax, ymax)) return;

  int bw = xmax - xmin + 1;   // largeur  du bounding box
  int bh = ymax - ymin + 1;   // hauteur  du bounding box

  // Facteur d'échelle : on veut tenir dans 20x20 en conservant le ratio
  float scale = 20.0f / (float)((bw > bh) ? bw : bh);

  int scaledW = (int)(bw * scale + 0.5f);
  int scaledH = (int)(bh * scale + 0.5f);

  // Calcul du centroïde (centre de masse des pixels) de l'image mise à l'échelle
  float cx = 0.0f, cy = 0.0f, wsum = 0.0f;

  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      if(src[y][x] > 0.05f){
        float sx = (x - xmin) * scale;
        float sy = (y - ymin) * scale;
        cx   += sx * src[y][x];
        cy   += sy * src[y][x];
        wsum += src[y][x];
      }
    }
  }

  if(wsum < 1e-6f) return;

  cx /= wsum;
  cy /= wsum;

  // Décalage pour centrer le centroïde sur (14, 14)
  int offX = (int)(14.0f - cx + 0.5f);
  int offY = (int)(14.0f - cy + 0.5f);

  // Rééchantillonnage bilinéaire : on parcourt la dst et on cherche
  // le pixel correspondant dans src
  for(int dy = 0; dy < GRID_SIZE; dy++){
    for(int dx = 0; dx < GRID_SIZE; dx++){

      // Coordonnées dans l'espace source
      float sx = (dx - offX) / scale + xmin;
      float sy = (dy - offY) / scale + ymin;

      if(sx < 0 || sx >= GRID_SIZE - 1 || sy < 0 || sy >= GRID_SIZE - 1) continue;

      // Interpolation bilinéaire
      int x0 = (int)sx,  y0 = (int)sy;
      int x1 = x0 + 1,   y1 = y0 + 1;
      float tx = sx - x0, ty = sy - y0;

      float v = src[y0][x0] * (1-tx) * (1-ty)
              + src[y0][x1] *    tx  * (1-ty)
              + src[y1][x0] * (1-tx) *    ty
              + src[y1][x1] *    tx  *    ty;

      dst[dy][dx] = (v > 1.0f) ? 1.0f : v;
    }
  }
}


// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  display.init();
  display.startWrite();
  display.fillScreen(TFT_BLACK);

  for(int y = 0; y < GRID_SIZE; y++)
    for(int x = 0; x < GRID_SIZE; x++)
      grid[y][x] = 0.0f;
}


// =====================================================
// LOOP : dessin tactile
// =====================================================
void loop() {

  lgfx::touch_point_t tp[1];
  int nums = display.getTouchRaw(tp, 1);

  static bool drawing = false;

  if(nums) {

    drawing = true;

    int x = tp[0].x;
    int y = tp[0].y;

    display.fillCircle(x, y, 4, TFT_WHITE);

    int gx = map(x, 0, 320, 0, 27);
    int gy = map(y, 0, 240, 0, 27);

    if(gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE)
      grid[gy][gx] = 1.0f;

    for(int dy = -1; dy <= 1; dy++){
      for(int dx = -1; dx <= 1; dx++){
        int nx = gx + dx;
        int ny = gy + dy;
        if(nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE){
          if(dx != 0 || dy != 0){
            if(grid[ny][nx] < 0.5f)
              grid[ny][nx] = 0.5f;
          }
        }
      }
    }

  }
  else if(drawing) {

    printGrid();
    runCNN();

    drawing = false;

    display.fillScreen(TFT_BLACK);

    for(int y = 0; y < GRID_SIZE; y++)
      for(int x = 0; x < GRID_SIZE; x++)
        grid[y][x] = 0.0f;
  }
}


// =====================================================
// EXECUTION CNN + PREPROCESSING COMPLET
// =====================================================
void runCNN() {

  // En ajoutant 'static', ces variables ne saturent plus la pile (Stack)
  static float blurred[GRID_SIZE][GRID_SIZE];
  static float preprocessed[GRID_SIZE][GRID_SIZE];
  static input_t input;
  static output_t output; // Celle-ci est petite, on peut la laisser

  // 1. Lissage
  gaussianBlur(grid, blurred);

  // 2. Centrage + mise à l'échelle
  centerAndScale(blurred, preprocessed);

  // 3. Préparation de l'entrée CNN
  for(int y = 0; y < 28; y++) {
    for(int x = 0; x < 28; x++) {
      input[y][x][0] = preprocessed[y][x];
    }
  }

  // 4. Inférence
  cnn(input, output);

  // 5. Résultats
  float probs[10];
  softmax(output, probs, 10);

  int predicted = 0;
  float maxVal  = probs[0];
  for(int i = 1; i < 10; i++){
    if(probs[i] > maxVal){
      maxVal = probs[i];
      predicted = i;
    }
  }

  // Affichage des résultats
  Serial.println("=========== RESULTAT CNN ===========");
  for(int i = 0; i < 10; i++){
    Serial.printf("%d : %.4f %%\n", i, probs[i] * 100);
  }
  Serial.printf("Chiffre reconnu : %d\n", predicted);
  Serial.println("====================================");
}
// =====================================================
// DEBUG : affichage grille
// =====================================================
void printGrid() {
  Serial.println("============= GRID =============");
  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      Serial.print(grid[y][x], 1);
      Serial.print(" ");
    }
    Serial.println();
  }
  Serial.println("================================");
}