#include <M5GFX.h>
#include <../CNN/mnist_float32.h>
#include <math.h>

M5GFX display;

#define GRID_SIZE 28

// Variables globales
float grid[GRID_SIZE][GRID_SIZE];
unsigned long lastPredictionTime = 0;
bool showResult = false;

// Prototypes
void runCNN();
void softmax(float *input, float *output, int size);
void gaussianBlur3x3(float src[GRID_SIZE][GRID_SIZE], float dst[GRID_SIZE][GRID_SIZE]);
void centerAndScale(float src[GRID_SIZE][GRID_SIZE], float dst[GRID_SIZE][GRID_SIZE]);
bool getBoundingBox(float src[GRID_SIZE][GRID_SIZE], int &xmin, int &ymin, int &xmax, int &ymax);

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  display.init();
  display.startWrite();
  display.fillScreen(TFT_BLACK);

  memset(grid, 0, sizeof(grid));
  Serial.println("Système prêt - Contraste élevé");
}

// =====================================================
// LOOP : Dessin tactile
// =====================================================
void loop() {
  lgfx::touch_point_t tp[1];
  int nums = display.getTouchRaw(tp, 1);
  static bool drawing = false;

  if (nums) {
    if (showResult) {
      display.fillScreen(TFT_BLACK);
      memset(grid, 0, sizeof(grid));
      showResult = false;
    }

    drawing = true;
    int x = tp[0].x;
    int y = tp[0].y;

    display.fillCircle(x, y, 5, TFT_WHITE);

    int gx = map(x, 0, 320, 0, 27);
    int gy = map(y, 0, 240, 0, 27);

    if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
      grid[gy][gx] = 1.0f;
      // Épaississement léger pour assurer la continuité du trait
      if (gx + 1 < 28) grid[gy][gx + 1] = std::max(grid[gy][gx + 1], 0.6f);
      if (gx - 1 >= 0) grid[gy][gx - 1] = std::max(grid[gy][gx - 1], 0.6f);
      if (gy + 1 < 28) grid[gy + 1][gx] = std::max(grid[gy + 1][gx], 0.6f);
      if (gy - 1 >= 0) grid[gy - 1][gx] = std::max(grid[gy - 1][gx], 0.6f);
    }
  } 
  else if (drawing) {
    runCNN();
    drawing = false;
    showResult = true;
    lastPredictionTime = millis();
  }

  if (showResult && (millis() - lastPredictionTime > 2500)) {
    display.fillScreen(TFT_BLACK);
    memset(grid, 0, sizeof(grid));
    showResult = false;
  }
}

// =====================================================
// EXECUTION CNN + TRAITEMENT IMAGE
// =====================================================
void runCNN() {
  static float blurred[GRID_SIZE][GRID_SIZE];
  static float preprocessed[GRID_SIZE][GRID_SIZE];
  static input_t input;
  static output_t output;

  // 1. Preprocessing
  gaussianBlur3x3(grid, blurred);
  centerAndScale(blurred, preprocessed);

  // --- NOUVEAU : RENFORCEMENT DU NOIR ET NETTOYAGE ---
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      float v = preprocessed[y][x];
      
      if (v > 0.35f) {
        // Courbe de contraste : les valeurs hautes deviennent très hautes (noir pur)
        v = powf(v, 0.7f) * 1.1f; 
      } else {
        // On coupe les gris trop clairs pour un dégradé plus net
        v = 0.0f; 
      }
      preprocessed[y][x] = (v > 1.0f) ? 1.0f : v;
    }
  }
  // ----------------------------------------------------

  // 2. Préparation Input
  for(int y = 0; y < 28; y++) {
    for(int x = 0; x < 28; x++) {
      input[y][x][0] = preprocessed[y][x];
    }
  }

  // --- MESURE DU TEMPS D'INFÉRENCE ---
  unsigned long startTime = micros(); // Début du chrono
  
  cnn(input, output); // Appel du modèle IA
  
  unsigned long endTime = micros();   // Fin du chrono
  unsigned long inferenceTime = endTime - startTime; // Temps en microsecondes
  // ------------------------------------

  // 3. Softmax & Résultat
  float probs[10];
  softmax(output, probs, 10);

  int predicted = 0;
  float maxVal = probs[0];
  for(int i = 1; i < 10; i++){
    if(probs[i] > maxVal) { maxVal = probs[i]; predicted = i; }
  }

  // 4. Envoi des données vers Python (on ajoute le temps à la fin)
  // Format : RES:Chiffre,Confiance,TempsInferenceMs
  Serial.print("MAT:"); // Envoi matrice (inchangé)
  for(int y=0; y<28; y++) for(int x=0; x<28; x++) {
    Serial.print(preprocessed[y][x], 1);
    if(x<27 || y<27) Serial.print(",");
  }
  Serial.println();

  Serial.printf("RES:%d,%.2f,%.3f\n", predicted, maxVal * 100, (float)inferenceTime / 1000.0f);

  // 5. Affichage sur l'écran du M5
  display.fillScreen(TFT_BLACK);
  display.setTextColor(TFT_GREEN, TFT_BLACK); 
  display.setTextSize(2);
  display.setCursor(10, 10);
  display.printf("ID: %d (%.1f%%)", predicted, maxVal * 100);
  
  display.setCursor(10, 40);
  display.setTextColor(TFT_WHITE, TFT_BLACK);
  display.printf("IA: %.2f ms", (float)inferenceTime / 1000.0f);
}

// =====================================================
// FONCTIONS TECHNIQUES
// =====================================================

void gaussianBlur3x3(float src[GRID_SIZE][GRID_SIZE], float dst[GRID_SIZE][GRID_SIZE]) {
  // Noyau 3x3 plus compact pour garder un coeur de trait sombre
  static const float kernel[3][3] = {
    { 0.075f, 0.124f, 0.075f },
    { 0.124f, 0.204f, 0.124f },
    { 0.075f, 0.124f, 0.075f }
  };

  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      float acc = 0.0f;
      for(int ky = -1; ky <= 1; ky++){
        for(int kx = -1; kx <= 1; kx++){
          int nx = x + kx;
          int ny = y + ky;
          if(nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE){
            acc += src[ny][nx] * kernel[ky + 1][kx + 1];
          }
        }
      }
      // On booste légèrement le résultat pour compenser la perte d'intensité du flou
      float v = acc * 1.2f; 
      dst[y][x] = (v > 1.0f) ? 1.0f : v;
    }
  }
}

void centerAndScale(float src[GRID_SIZE][GRID_SIZE], float dst[GRID_SIZE][GRID_SIZE]) {
  memset(dst, 0, sizeof(float) * GRID_SIZE * GRID_SIZE);
  int xmin, ymin, xmax, ymax;
  if (!getBoundingBox(src, xmin, ymin, xmax, ymax)) return;

  int bw = xmax - xmin + 1, bh = ymax - ymin + 1;
  float scale = 20.0f / (float)((bw > bh) ? bw : bh);

  float cx = 0, cy = 0, wsum = 0;
  for (int y = ymin; y <= ymax; y++) {
    for (int x = xmin; x <= xmax; x++) {
      if (src[y][x] > 0.1f) {
        cx += (x - xmin) * scale * src[y][x];
        cy += (y - ymin) * scale * src[y][x];
        wsum += src[y][x];
      }
    }
  }
  if (wsum < 0.001f) return;
  cx /= wsum; cy /= wsum;

  int offX = (int)(14.0f - cx), offY = (int)(14.0f - cy);

  for (int dy = 0; dy < GRID_SIZE; dy++) {
    for (int dx = 0; dx < GRID_SIZE; dx++) {
      float sx = (dx - offX) / scale + xmin, sy = (dy - offY) / scale + ymin;
      if (sx >= 0 && sx < GRID_SIZE - 1 && sy >= 0 && sy < GRID_SIZE - 1) {
        int x0 = (int)sx, y0 = (int)sy;
        float tx = sx - x0, ty = sy - y0;
        dst[dy][dx] = src[y0][x0] * (1 - tx) * (1 - ty) + src[y0][x0 + 1] * tx * (1 - ty) +
                      src[y0 + 1][x0] * (1 - tx) * ty + src[y0 + 1][x0 + 1] * tx * ty;
      }
    }
  }
}

bool getBoundingBox(float src[GRID_SIZE][GRID_SIZE], int &xmin, int &ymin, int &xmax, int &ymax) {
  xmin = ymin = GRID_SIZE; xmax = ymax = -1;
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      if (src[y][x] > 0.1f) {
        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
      }
    }
  }
  return (xmax >= xmin);
}

void softmax(float *input, float *output, int size) {
  float maxVal = input[0];
  for (int i = 1; i < size; i++) if (input[i] > maxVal) maxVal = input[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    output[i] = expf(input[i] - maxVal);
    sum += output[i];
  }
  for (int i = 0; i < size; i++) output[i] /= sum;
}