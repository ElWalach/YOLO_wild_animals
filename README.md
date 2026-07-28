# YOLO Wild Animals 🦌🐗🐅

System detekcji dzikich zwierząt oparty na architekturze **YOLOv8**, zrealizowany w ramach projektu z przedmiotu *Sztuczna Inteligencja*.

Model automatycznie rozpoznaje i lokalizuje **17 gatunków dzikich zwierząt** na zdjęciach wykonanych w naturalnym środowisku leśnym i górskim Azji Wschodniej, m.in. jelenia sika, sarnę, dzika, lisa rudego, jenota azjatyckiego, tygrysa amurskiego, lamparta i innych.

## 📋 Opis projektu

Celem projektu było opracowanie i przeanalizowanie systemu detekcji obiektów opartego na YOLOv8 (biblioteka Ultralytics, backend PyTorch), zweryfikowanie jego skuteczności w trudnych warunkach wizyjnych typowych dla fotografii przyrodniczej (kamuflaż, zmienne oświetlenie, złożone tło, częściowe zasłonięcia) oraz zbadanie wpływu kluczowych hiperparametrów na jakość detekcji metodą Grid Search.

Zbadano dwa wymiary konfiguracji:
- **wariant modelu**: YOLOv8n (Nano) vs. YOLOv8s (Small)
- **rozmiar obrazu wejściowego**: 416×416 vs. 640×640 pikseli

co dało łącznie 4 eksperymenty treningowe.

## 🗂️ Zbiór danych

- Źródło: publiczny zbiór **Wild Animals** (Roboflow, `personal-exflz/wild-animals-s8rz2`, wersja 1)
- **9697** obrazów, ~9200 adnotacji w podzbiorze treningowym
- **17 klas**: m.in. Hare, RaccoonDog, SikaDeer, RoeDeer, WildBoar, RedFox, AmurTiger, Leopard, LeopardCat, BlackBear, Badger, MuskDeer, Y-T-Marten, Sable, Dog, Cow
- Format adnotacji: YOLOv8 (znormalizowane bounding boxy)
- Zdjęcia z kamer pułapkowych oraz aparatów obsługiwanych ręcznie, o zróżnicowanych warunkach oświetleniowych i porze dnia
- Wyraźne niezbalansowanie klas (stosunek klasy najliczniejszej do najmniej licznej ok. **12,7:1**)

## 🧠 Metodologia

- Transfer learning z wagami pretrenowanymi na zbiorze **COCO**
- Augmentacja danych: Mosaic, Horizontal Flip, Scale, Translate, Blur, MedianBlur, ToGray, CLAHE, Random Erasing (Albumentations)
- Optymalizator **AdamW** dobierany automatycznie (`optimizer=auto`), harmonogram warm-up + linear decay
- Trening: 50 epok, `close_mosaic=10`, AutoBatch
- Środowisko: **Kaggle**, GPU **Tesla T4** (14 GB VRAM)
- Ewaluacja na wydzielonym zbiorze testowym (968 obrazów, 1016 instancji obiektów)

## 📊 Wyniki

Najlepsza konfiguracja — **YOLOv8s, imgsz=416**:

| Metryka | Wartość |
|---|---|
| mAP@50 | 0,977 |
| mAP@50-95 | 0,848 |
| Precision | 0,975 |
| Recall | 0,939 |
| Prędkość inferencji | ~294 fps (GPU Tesla T4) |

Wszystkie 4 przetestowane konfiguracje osiągnęły **mAP@50 > 97%**.

### Kluczowe wnioski

- **Nano vs. Small**: różnica jakości jest minimalna (+0,35 p.p. mAP@50 dla Small), mimo 3,7× większej liczby parametrów — dla zastosowań na urządzeniach brzegowych model **Nano** jest zdecydowanie lepszym kompromisem.
- **416 vs. 640 px**: mniejszy rozmiar obrazu wypada konsekwentnie lepiej we wszystkich metrykach, co wynika z charakterystyki zbioru (obiekty stosunkowo duże w kadrze) oraz większej liczby aktualizacji wag przy tym samym budżecie epok.
- Klasy nielicznie reprezentowane (**Cow, Dog, Y-T-Marten**) osiągają zauważalnie gorsze wyniki per-klasa, co potwierdza wpływ niezbalansowania danych na jakość detekcji rzadkich gatunków.

## 🛠️ Technologie

- Python, PyTorch
- Ultralytics YOLOv8
- Albumentations
- Kaggle (GPU Tesla T4)

## 📁 Zawartość repozytorium

- [`si-projekt.ipynb`](./si-projekt.ipynb) — pełny notebook z przygotowaniem danych, treningiem, ewaluacją i analizą wyników wszystkich czterech eksperymentów.

## 🚀 Uruchomienie

1. Otwórz `si-projekt.ipynb` w Jupyterze/Kaggle/Colab.
2. Zainstaluj zależności:
```bash
   pip install ultralytics albumentations
```
3. Pobierz zbiór danych **Wild Animals** z Roboflow (`personal-exflz/wild-animals-s8rz2`, wersja 1).
4. Uruchom komórki notebooka — trening, ewaluację i analizę wyników.
