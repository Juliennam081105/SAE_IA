#include <M5Unified.h>
#include <mnist_fixed_int16.h>
#include <math.h>

#define GRID_SIZE 28

// Variables globales
int16_t grid[GRID_SIZE][GRID_SIZE];
unsigned long lastPredictionTime = 0;
bool showResult = false;

// Prototypes
void runCNN();
void softmax(int16_t *input, float *output, int size);
void gaussianBlur3x3(int16_t src[GRID_SIZE][GRID_SIZE], int16_t dst[GRID_SIZE][GRID_SIZE]);
void centerAndScale(int16_t src[GRID_SIZE][GRID_SIZE], int16_t dst[GRID_SIZE][GRID_SIZE]);
bool getBoundingBox(int16_t src[GRID_SIZE][GRID_SIZE], int &xmin, int &ymin, int &xmax, int &ymax);

// =====================================================
// SETUP
// =====================================================
void setup() {
  auto cfg = M5.config();
  M5.begin(cfg); // Initialise Display, Touch, et Power (AXP2101)

  Serial.begin(115200);
  
  M5.Display.fillScreen(TFT_BLACK);
  memset(grid, 0, sizeof(grid));
  
  Serial.println("Système prêt - M5Unified");
}

// =====================================================
// LOOP : Dessin tactile
// =====================================================
void loop() {
  M5.update(); // Crucial pour rafraîchir les données tactile et batterie
  
  auto count = M5.Touch.getCount();
  static bool drawing = false;

  if (count > 0) {
    auto t = M5.Touch.getDetail(0); // Récupère les infos du premier point touché

    if (showResult) {
      M5.Display.fillScreen(TFT_BLACK);
      memset(grid, 0, sizeof(grid));
      showResult = false;
    }

    drawing = true;
    
    // On dessine directement via M5.Display
    M5.Display.fillCircle(t.x, t.y, 5, TFT_WHITE);

    // Map les coordonnées vers la grille 28x28
    int gx = map(t.x, 0, M5.Display.width(), 0, 27);
    int gy = map(t.y, 0, M5.Display.height(), 0, 27);

    if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
      grid[gy][gx] = 512; 
      
      // Épaississement (Q9 format)
      if (gx + 1 < 28) grid[gy][gx + 1] = max((int)grid[gy][gx+1], 307);
      if (gx - 1 >= 0) grid[gy][gx - 1] = max((int)grid[gy][gx-1], 307);
      if (gy + 1 < 28) grid[gy + 1][gx] = max((int)grid[gy+1][gx], 307);
      if (gy - 1 >= 0) grid[gy - 1][gx] = max((int)grid[gy-1][gx], 307);
    }
  } 
  else if (drawing) {
    runCNN();
    drawing = false;
    showResult = true;
    lastPredictionTime = millis();
  }

  // Auto-clear après 2.5 secondes
  if (showResult && (millis() - lastPredictionTime > 2500)) {
    M5.Display.fillScreen(TFT_BLACK);
    memset(grid, 0, sizeof(grid));
    showResult = false;
  }
}

// =====================================================
// EXECUTION CNN + TRAITEMENT IMAGE
// =====================================================
void runCNN() {
  static int16_t blurred[GRID_SIZE][GRID_SIZE];
  static int16_t preprocessed[GRID_SIZE][GRID_SIZE];
  static input_t input;
  static dense_1_output_type output; 
  
  // 1. Preprocessing
  gaussianBlur3x3(grid, blurred);
  centerAndScale(blurred, preprocessed);

  // Renforcement contraste
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      int16_t v = preprocessed[y][x];
      if (v > 179) {
        int32_t boosted = (int32_t)v * 580 / 512;
        preprocessed[y][x] = (boosted > 512) ? 512 : (int16_t)boosted;
      } else {
        preprocessed[y][x] = 0; 
      }
      input[y][x][0] = preprocessed[y][x];
    }
  }

  // 2. Inférence
  unsigned long startTime = micros();
  cnn(input, output);
  unsigned long inferenceTime = micros() - startTime;

  // 3. Softmax
  float probs[10];
  softmax(output, probs, 10);

  int predicted = 0;
  float maxVal = probs[0];
  for(int i = 1; i < 10; i++){
    if(probs[i] > maxVal) { maxVal = probs[i]; predicted = i; }
  }

  // 4. Envoi Serial (pour Python)
  Serial.print("MAT:");
  for(int y=0; y<28; y++) {
    for(int x=0; x<28; x++) {
      Serial.print((float)preprocessed[y][x] / 512.0f, 1);
      if(x<27 || y<27) Serial.print(",");
    }
  }
  Serial.println();
  Serial.printf("RES:%d,%.2f,%.3f\n", predicted, maxVal * 100, (float)inferenceTime / 1000.0f);

  // 5. Affichage M5Unified
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setTextColor(TFT_GREEN, TFT_BLACK); 
  M5.Display.setTextSize(2);
  M5.Display.setCursor(10, 10);
  M5.Display.printf("ID: %d (%.1f%%)", predicted, maxVal * 100);
  
  M5.Display.setCursor(10, 50);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.printf("IA: %.2f ms", (float)inferenceTime / 1000.0f);

  // Gestion Batterie Corrigée
  int batPct = M5.Power.getBatteryLevel();
  bool isCharging = M5.Power.isCharging();

  M5.Display.setCursor(10, 100);
  M5.Display.setTextSize(1);
  M5.Display.setTextColor(TFT_YELLOW, TFT_BLACK);

  if (isCharging) {
    M5.Display.printf("Batterie : %d%% (En charge)", batPct);
  } else {
    M5.Display.printf("Batterie : %d%% (Sur batterie)", batPct);
  }
}

// --- Garde tes fonctions techniques (gaussianBlur3x3, centerAndScale, etc.) identiques ---


// =====================================================
// FONCTIONS TECHNIQUES
// =====================================================

void gaussianBlur3x3(int16_t src[GRID_SIZE][GRID_SIZE], int16_t dst[GRID_SIZE][GRID_SIZE]) {
  // Noyau converti en Q12 (plus de précision pour les petits coefficients)
  // Somme des coefficients = 4096 (soit 1.0 en Q12)
  static const int32_t kernel[3][3] = {
    { 307, 508, 307 },
    { 508, 836, 508 },
    { 307, 508, 307 }
  };

  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      int32_t acc = 0;
      for(int ky = -1; ky <= 1; ky++){
        for(int kx = -1; kx <= 1; kx++){
          int nx = x + kx;
          int ny = y + ky;
          if(nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE){
            acc += (int32_t)src[ny][nx] * kernel[ky + 1][kx + 1];
          }
        }
      }
      // On divise par 4096 (>> 12) pour annuler le format du noyau
      // Puis on applique le boost de 1.2x (6/5)
      int32_t v = (acc >> 12) * 6 / 5;
      
      // Saturation à 1.0 (qui est 512 en Q9)
      if (v > 512) v = 512;
      dst[y][x] = (int16_t)v;
    }
  }
}

void centerAndScale(int16_t src[GRID_SIZE][GRID_SIZE], int16_t dst[GRID_SIZE][GRID_SIZE]) {
  memset(dst, 0, sizeof(int16_t) * GRID_SIZE * GRID_SIZE);
  int xmin, ymin, xmax, ymax;
  if (!getBoundingBox(src, xmin, ymin, xmax, ymax)) return;

  int bw = xmax - xmin + 1, bh = ymax - ymin + 1;
  float scale = 20.0f / (float)((bw > bh) ? bw : bh);

  float cx = 0, cy = 0;
  long wsum = 0;
  for (int y = ymin; y <= ymax; y++) {
    for (int x = xmin; x <= xmax; x++) {
      if (src[y][x] > 51) { // seuil 0.1
        cx += (float)(x - xmin) * scale * src[y][x];
        cy += (float)(y - ymin) * scale * src[y][x];
        wsum += src[y][x];
      }
    }
  }
  if (wsum == 0) return;
  cx /= (float)wsum; cy /= (float)wsum;

  int offX = (int)(14.0f - cx), offY = (int)(14.0f - cy);

  for (int dy = 0; dy < GRID_SIZE; dy++) {
    for (int dx = 0; dx < GRID_SIZE; dx++) {
      float sx = (float)(dx - offX) / scale + xmin;
      float sy = (float)(dy - offY) / scale + ymin;
      
      if (sx >= 0 && sx < GRID_SIZE - 1 && sy >= 0 && sy < GRID_SIZE - 1) {
        int x0 = (int)sx, y0 = (int)sy;
        float tx = sx - x0, ty = sy - y0;
        
        // Interpolation bilinéaire
        float res = (float)src[y0][x0] * (1.0f - tx) * (1.0f - ty) +
                    (float)src[y0][x0 + 1] * tx * (1.0f - ty) +
                    (float)src[y0 + 1][x0] * (1.0f - tx) * ty +
                    (float)src[y0 + 1][x0 + 1] * tx * ty;
        dst[dy][dx] = (int16_t)res;
      }
    }
  }
}

bool getBoundingBox(int16_t src[GRID_SIZE][GRID_SIZE], int &xmin, int &ymin, int &xmax, int &ymax) {
  xmin = ymin = GRID_SIZE; xmax = ymax = -1;
  int16_t threshold = 51; // 0.1 en Q9
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      if (src[y][x] > threshold) {
        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
      }
    }
  }
  return (xmax >= xmin);
}

void softmax(int16_t *input, float *output, int size) {
  // On trouve le score max pour la stabilité numérique
  int16_t maxVal = input[0];
  for (int i = 1; i < size; i++) if (input[i] > maxVal) maxVal = input[i];
  
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    // On convertit le score int16_t en float avant l'exp
    // Note: les scores de sortie n'ont plus forcément l'échelle Q9
    output[i] = expf((float)(input[i] - maxVal) * 0.1f); 
    sum += output[i];
  }
  for (int i = 0; i < size; i++) output[i] /= sum;
}