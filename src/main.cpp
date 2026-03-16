#include <M5GFX.h>
#include <mnist_float32.h>
#include <math.h>

M5GFX display;

#define GRID_SIZE 28
int grid[GRID_SIZE][GRID_SIZE];

input_t input;
output_t output;

// --- FONCTIONS DE CALCUL ---

// Transforme les scores bruts en probabilités (0.0 à 1.0)
void applySoftmax(float* data, int size) {
    float maxVal = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > maxVal) maxVal = data[i];
    }

    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        data[i] = exp(data[i] - maxVal); 
        sum += data[i];
    }

    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

void clearGrid() {
    for(int y=0; y<GRID_SIZE; y++) {
        for(int x=0; x<GRID_SIZE; x++) {
            grid[y][x] = 0;
        }
    }
}

// Fonction avancée pour recentrer le dessin dans la grille 28x28
void gridToInputWithCentering() {
    int minX = GRID_SIZE, maxX = 0, minY = GRID_SIZE, maxY = 0;
    bool empty = true;

    // 1. Trouver la "bounding box" du chiffre dessiné
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if (grid[y][x] == 1) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                empty = false;
            }
        }
    }

    // Initialiser l'entrée à 0 (noir)
    for(int y=0; y<28; y++) for(int x=0; x<28; x++) input[y][x][0] = 0;

    if (empty) return;

    // 2. Calculer le décalage pour centrer
    int width = maxX - minX + 1;
    int height = maxY - minY + 1;
    int offsetX = (GRID_SIZE - width) / 2;
    int offsetY = (GRID_SIZE - height) / 2;

    // 3. Copier le chiffre au centre de l'input
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (grid[minY + y][minX + x] == 1) {
                input[offsetY + y][offsetX + x][0] = 1.0f;
            }
        }
    }
}

// --- SETUP ET LOOP ---

void setup() {
    Serial.begin(115200);
    display.init();
    display.setRotation(1); // Ajustez selon votre écran
    display.fillScreen(TFT_BLACK);
    clearGrid();
    Serial.println("Prêt à détecter !");
}

void loop() {
    lgfx::touch_point_t tp[1];
    int n = display.getTouch(tp, 1);
    static bool drawing = false;

    if (n) {
        drawing = true;
        int x = tp[0].x;
        int y = tp[0].y;

        // Dessin sur l'écran (blanc sur noir)
        display.fillCircle(x, y, 6, TFT_WHITE);

        // Mapping vers la grille
        int gx = map(x, 0, display.width(), 0, 27);
        int gy = map(y, 0, display.height(), 0, 27);

        // Dessin épais 3x3 dans la grille interne
        for(int dy=-1; dy<=1; dy++) {
            for(int dx=-1; dx<=1; dx++) {
                int nx = gx + dx;
                int ny = gy + dy;
                if(nx >= 0 && nx < 28 && ny >= 0 && ny < 28) grid[ny][nx] = 1;
            }
        }
    } 
    else if (drawing) {
        drawing = false;
        Serial.println("\n--- Analyse en cours ---");

        // Préparation des données avec centrage
        gridToInputWithCentering();

        // Exécution du modèle
        cnn(input, output);

        // Application du Softmax pour obtenir des probabilités
        applySoftmax(output, 10);

        // Trouver le meilleur score
        int best = 0;
        float maxProb = 0;
        for (int i = 0; i < 10; i++) {
            Serial.printf("[%d]: %.2f%%\n", i, output[i] * 100.0);
            if (output[i] > maxProb) {
                maxProb = output[i];
                best = i;
            }
        }

        Serial.printf(">> DETECTION : %d (Confiance: %.1f%%)\n", best, maxProb * 100.0);

        delay(100);
        display.fillScreen(TFT_BLACK);
        clearGrid();
    }
}