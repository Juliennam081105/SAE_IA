#include <M5GFX.h>
#include <mnist_float32.h>

M5GFX display;

#define GRID_SIZE 28

int grid[GRID_SIZE][GRID_SIZE];

void printGrid();  // prototype pour afficher la grille

void setup() {
  Serial.begin(115200);
  Serial.println("Programme démarre");

  display.init();
  display.startWrite();

  // Fond noir
  display.fillScreen(TFT_BLACK);

  // Initialiser la grille
  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      grid[y][x] = 0;
    }
  }

  Serial.println("Grille initialisée");
}

void loop() {
  lgfx::touch_point_t tp[1];
  int n = display.getTouch(tp, 1);

  static bool drawing = false; // indique si un dessin est en cours

  if(n) { // le doigt est sur l'écran
    drawing = true;

    int x = tp[0].x;
    int y = tp[0].y;

    // dessiner sur l'écran en blanc
    display.fillCircle(x, y, 4, TFT_WHITE);

    // conversion écran -> grille 28x28
    int gx = map(x, 0, 320, 0, 27);
    int gy = map(y, 0, 240, 0, 27);

    // --- remplissage "plus gras" ---
    for(int dy = -1; dy <= 1; dy++){
      for(int dx = -1; dx <= 1; dx++){
        int nx = gx + dx;
        int ny = gy + dy;

        if(nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE){
          grid[ny][nx] = 1;
        }
      }
    }

  } else if(drawing) { 
    // le doigt a été retiré → dessin terminé
    drawing = false;

    Serial.println("===== FIN DU DESSIN =====");
    printGrid();  // afficher la grille finale

    // réinitialiser écran et matrice
    display.fillScreen(TFT_BLACK);
    for(int y = 0; y < GRID_SIZE; y++){
      for(int x = 0; x < GRID_SIZE; x++){
        grid[y][x] = 0;
      }
    }

    Serial.println("===== GRILLE REMISE A ZERO =====");
    printGrid();  // afficher la grille vide
  }
}

void printGrid() {
  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      Serial.print(grid[y][x]);
      Serial.print(" ");
    }
    Serial.println();
  }
  Serial.println("----------------------------");
}