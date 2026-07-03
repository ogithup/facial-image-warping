# Facial Image Warping

Bu repo, **Facial Image Warping, Aging, and Expression Transformation** projesinin ilk on sprintinin calisan temelini icerir. Su ana kadar odak noktasi, yuzu degistirmekten once veriyi **dogru almak**, **DSP acisindan standardize etmek**, **yuz bolgesini ayirmak**, **yuzu landmark noktalariyla temsil etmek**, **geometric warping** uygulamak ve son olarak **frekans alaninda analiz etmek** oldu.

Mevcut durumda tamamlanan katmanlar:

1. `Sprint 1` - Image input ve preprocessing
2. `Sprint 2` - Face detection ve face crop
3. `Sprint 3` - Facial landmark detection ve landmark export
4. `Sprint 4` - Geometric facial image warping
5. `Sprint 5` - Frequency domain DSP analysis
6. `Sprint 6` - Aging and de-aging simulation
7. `Sprint 7` - Quantitative evaluation module
8. `Sprint 8` - Streamlit GUI
9. `Sprint 9` - Real-time webcam integration
10. `Sprint 10` - Advanced AI-based expression transfer

Henuz tamamlanmayan katmanlar:

- GAN/diffusion tabanli generative expression transfer
- Full production-grade browser webcam streaming

## DSP Mantigi

Bu projede goruntu, klasik bir fotograf dosyasi gibi degil, **2 boyutlu sayisal sinyal** olarak ele alinir. Kurulan akis su mantiga dayanir:

`User Input -> Image Acquisition -> Preprocessing -> ROI Extraction -> Landmark Representation -> Geometric Warping -> Frequency Analysis -> Future DSP Stages`

Bu zincirde su ana kadar yapilan isler:

- **Image acquisition**: giris dosyasini dogrulama ve bellege alma
- **Preprocessing**: renk uzayi donusumu, grayscale, resize, normalize etme
- **ROI extraction**: tum goruntu yerine yalnizca yuz bolgesini secme
- **Geometric representation**: yuzu landmark ile parametrik olarak ifade etme
- **Triangle warping**: landmark displacement ile affine triangle deformasyonu
- **Frequency analysis**: uzamsal alandan frekans alanina gecip spektral enerji olcme
- **Quantitative evaluation**: donusum farkini MSE, PSNR ve SSIM ile olcme
- **Interactive UI**: ayni pipelinei Streamlit arayuzu uzerinden parametre kontrollu calistirma

Bu sayede aging, de-aging, kalite analizi, webcam denemeleri ve referans ifadeye dayali transfer islemleri dogrudan ham goruntuye degil, temizlenmis ve anlamli hale getirilmis yuz verisine uygulanir.

## Su Ana Kadarki Mimari

### Ana moduller

1. `input_module.py`
   Giris dosyasini dogrular, goruntuyu yukler, metadata uretir.
2. `preprocessing.py`
   `BGR/RGB/grayscale`, resize, normalization, histogram ve processed image uretir.
3. `face_detection.py`
   OpenCV Haar Cascade ile yuz bulur, bounding box cizer, yuz crop alir.
4. `landmark_detection.py`
   MediaPipe Face Mesh ile landmark cikarir, JSON/CSV export yapar.
5. `geometric_warping.py`
   Target landmark uretir, Delaunay triangulation kurar, triangle affine warp uygular.
6. `fourier_analysis.py`
   FFT, magnitude spectrum, spektral enerji ve CSV export uretir.
7. `evaluation.py`
   MSE, PSNR, SSIM, difference visualization ve CSV export uretir.
8. `visualization.py`
   Pipeline sonuc ozetlerini ve UI destek verilerini toplar.
9. `app.py`
   Sprint bazli pipeline giris noktalarini saglar.
10. `streamlit_app.py`
   Upload, webcam capture, transfer, spectrum ve metric tablo arayuzunu saglar.
11. `expression_transfer.py`
   Referans yuzden ifade geometrisini source yuze aktarir.
12. `webcam_demo.py`
   OpenCV tabanli gercek zamanli webcam donusum demosunu calistirir.

### Pipeline giris noktalari

- `run_preprocessing_pipeline(...)`
- `run_face_detection_pipeline(...)`
- `run_landmark_pipeline(...)`
- `run_expression_warp_pipeline(...)`
- `run_frequency_analysis_pipeline(...)`
- `run_aging_pipeline(...)`
- `run_deaging_pipeline(...)`
- `run_analysis_pipeline(...)`
- `run_reference_expression_transfer_pipeline(...)`

## Ornek Akis

Asagidaki ornek, `samples/test_face_1.jpg` uzerinden sistemin su ana kadar neleri urettigini gosterir.

### 1. Ornek giris yuzu

Bu gorsel sistemin aldigi ham inputtur. Bu asamada henuz sinyal secimi yoktur; arka plan ve yuz birlikte islenir.

![Sample Face](docs/images/sample-face.jpg)

### 2. Preprocessed face

Bu cikti preprocessing adimindan gelir:

- goruntu okunur
- standart cozunurluge tasinir
- normalize edilir
- sonraki asamalarin tutarli calismasi icin standardize edilir

Bu adimin DSP karsiligi, giris sinyalini ortak bir olcege cekmek ve sonraki islemlere uygun hale getirmektir.

![Preprocessed Face](docs/images/preprocessed-face.png)

### 3. Histogram ciktisi

Histogram, goruntu yogunluk dagilimini gosterir. Bu, sinyalin enerji dagilimi kadar guclu bir analiz degildir ama preprocessing asamasinda parlaklik ve kontrast karakterini anlamak icin temel bir aractir.

![Histogram](docs/images/histogram.png)

### 4. Detected face

Bu asamada OpenCV Haar Cascade ile yuz bulunur ve bounding box cizilir.

Burada yapilan is:

- nesne tespiti
- koordinat uretimi
- ROI secimi icin yuz sinirini tanimlama

DSP acisindan bu adim, butun sinyalden sadece anlamli alt bolgeyi ayirmaya giden ilk adimdir.

![Detected Face](docs/images/detected-face.png)

### 5. Face crop

Bounding box kullanilarak yuz bolgesi kesilir ve ayri bir goruntu olarak normalize edilir.

Bu adim cok kritiktir cunku sonraki landmark, warping ve aging islemleri tum fotograf uzerinde degil, yalnizca bu ROI uzerinde yapilacaktir.

![Face Crop](docs/images/face-crop.png)

### 6. Landmark ciktisi

Sprint 3 ile MediaPipe Face Mesh uzerinden landmark uretimi eklenmistir. Calisan akis sonunda su dosyalar uretilir:

- `outputs/landmarks/<name>_landmarks.png`
- `outputs/landmarks/<name>_landmarks.json`
- `outputs/landmarks/<name>_landmarks.csv`

Asagidaki ornek gorsel, `test_face_1.jpg` icin uretilen gercek landmark overlay ciktisidir:

![Landmark Face](docs/images/landmarks-face.png)

Bu asama, geometric warping icin kritik veri uretir. Sonraki sprintlerde agiz, kas, goz, cene ve yanak deformasyonlari dogrudan bu landmark koordinatlari uzerinden yapilacaktir.

### 7. Geometric warping ciktisi

Sprint 4 ile landmark displacement tabanli geometric warping eklendi. Burada sistem:

- source landmarklari alir
- expression tipine gore target landmark uretir
- Delaunay triangulation kurar
- her triangle icin affine transform uygular
- triangle sonuclarini bosluk ve discontinuity olusturmadan birlestirir

Bu asama su kavramlari ogretir:

- affine transform
- triangle warping
- interpolation
- coordinate mapping
- mesh-based deformation

### 8. Frequency domain analysis

Sprint 5 ile goruntu uzamsal alandan frekans alanina tasinir. Burada:

- grayscale goruntu uzerinde `2D FFT` hesaplanir
- sifir frekans bileseni merkeze kaydirilir
- `log magnitude spectrum` elde edilir
- toplam, dusuk ve yuksek frekans enerjileri hesaplanir
- `high / low frequency ratio` bulunur
- sonuclar CSV olarak disa aktarilir

Bu asama su kavramlari ogretir:

- spatial domain
- frequency domain
- magnitude spectrum
- low / high frequency separation
- spectral energy analysis

## Landmark Koordinatlarinin Mantigi

MediaPipe, landmark noktalarini once **normalize koordinat** olarak verir:

- `x` degeri genislige gore `0-1`
- `y` degeri yukseklige gore `0-1`

Bu yaklasimin avantaji:

- farkli cozunurluklerde ayni yuz geometrisini tasimak kolaylasir
- model ekran pikseline degil goreli konuma konusur

Ama OpenCV cizimi ve crop islemleri icin bu koordinatlar piksele cevrilmelidir. Bu yuzden `landmark_detection.py` icinde normalize degerler mutlak piksele donusturulur.

## Su Ana Kadar Hangi Gereksinimler Tamamlandi

### Sprint 1

- JPG ve PNG yukleme
- dosya uzantisi ve varlik dogrulama
- `BGR -> RGB -> GRAYSCALE` donusumleri
- standart cozunurlukte resize
- piksel normalizasyonu
- histogram olusturma
- processed image kaydetme

### Sprint 2

- frontal face detection
- bounding box cizme
- detected face preview kaydetme
- crop alma
- crop'u normalize etme
- yuz bulunamazsa acik hata verme

### Sprint 3

- MediaPipe ile face mesh cikarma
- full landmark visualization
- region-based visualization
- JSON export
- CSV export
- landmark visualization toggle

### Sprint 4

- target landmark creation
- smile enhancement
- eyebrow raising
- lip widening
- face slimming
- Delaunay triangulation
- triangle-based affine warp
- before-after comparison kaydetme
- intensity kontrollu deformasyon

### Sprint 5

- grayscale frequency analysis
- 2D FFT
- FFT shift
- log magnitude spectrum
- side-by-side spectrum visualization
- total spectral energy
- low-frequency energy
- high-frequency energy
- high / low energy ratio
- CSV export

### Sprint 6

- high-frequency texture enhancement
- wrinkle-like detail simulation
- local contrast boost
- edge-preserving aging filter
- low-pass smoothing for de-aging
- bilateral filtering for skin softening
- adjustable intensity parameter
- outputs/aging/ icine kaydetme
- uygulanan filtreleri metinsel olarak donme

### Sprint 7

- MSE hesaplama
- PSNR hesaplama
- SSIM hesaplama
- image difference visualization
- evaluation metric CSV export
- original-transformed quantitative comparison

### Sprint 8

- Streamlit GUI
- image upload
- transformation selection
- intensity slider
- landmark toggle
- before-after display
- Fourier spectrum before-after display
- metric table
- CSV download

### Sprint 9

- OpenCV webcam capture
- real-time face crop preview
- real-time aging/de-aging and warping demo
- reference expression transfer in webcam mode
- keyboard-controlled demo exit

### Sprint 10

- reference image based expression transfer
- AI-assisted landmark-driven geometry transfer
- source-reference landmark visualization
- transfer region selection
- transfer explanation output

## Dosya Yapisi

```text
facial-image-warping/
|- app.py
|- pyproject.toml
|- README.md
|- docs/
|  |- images/
|- outputs/
|- samples/
|- src/
|  |- facial_image_warping/
|     |- face_detection.py
|     |- geometric_warping.py
|     |- fourier_analysis.py
|     |- input_module.py
|     |- landmark_detection.py
|     |- preprocessing.py
|     `- ...
`- tests/
```

## Kurulum

Bu proje icin surumler, "en yeni olan" mantigiyla degil, **koddaki API kullanimiyla uyumlu** olacak sekilde secildi.

Ozellikle:

- `opencv-python>=4.10,<5`
- `mediapipe==0.10.9`
- `numpy>=1.26,<2.0`
- `scikit-image>=0.24,<1`
- `streamlit>=1.36,<2`

Kurulum:

```bash
pip install -e ".[dev]"
```

## Calistirma

Terminal prompt su olmali:
`PS C:\facial-image-warping>`

Eger prompt baska yerdeyse, once:

```powershell
cd C:\facial-image-warping
```

### Sprint 1

```bash
python -c "from app import run_preprocessing_pipeline; r = run_preprocessing_pipeline('samples/test_face_1.jpg'); print(r['processed_image_path']); print(r['histogram_path'])"
```

### Sprint 2

```bash
python -c "from app import run_face_detection_pipeline; r = run_face_detection_pipeline('samples/test_face_1.jpg'); print(r['bounding_box']); print(r['cropped_face_path'])"
```

### Sprint 3

```bash
python -c "from app import run_landmark_pipeline; r = run_landmark_pipeline('samples/test_face_1.jpg', show_full_mesh=True, selected_regions=['eyes','nose','lips']); print(r['landmark_count']); print(r['json_path']); print(r['csv_path'])"
```

### Sprint 4

```bash
python -c "from app import run_expression_warp_pipeline; r = run_expression_warp_pipeline('samples/test_face_1.jpg', transformation='smile_enhancement', intensity=0.7); print(r['operation']); print(r['warped_image_path']); print(r['comparison_image_path'])"
```

### Sprint 5

```bash
python -c "from app import run_frequency_analysis_pipeline; r = run_frequency_analysis_pipeline('samples/test_face_1.jpg'); print(r['total_energy']); print(r['high_low_ratio']); print(r['spectrum_path']); print(r['csv_path'])"
```

### Sprint 6

```bash
python -c "from app import run_aging_pipeline; r = run_aging_pipeline('samples/test_face_1.jpg', intensity=0.7); print(r['mode']); print(r['intensity']); print(r['output_path']); print(r['filter_explanation'])"
```

```bash
python -c "from app import run_deaging_pipeline; r = run_deaging_pipeline('samples/test_face_1.jpg', intensity=0.4); print(r['mode']); print(r['intensity']); print(r['output_path']); print(r['filter_explanation'])"
```

### Sprint 7

```bash
python -c "from app import run_analysis_pipeline; r = run_analysis_pipeline('samples/test_face_1.jpg', transformation='smile_enhancement', intensity=0.6, show_landmarks=True); print(r['metrics']['mse']); print(r['metrics']['psnr']); print(r['metrics']['ssim']); print(r['metrics']['csv_path'])"
```

### Sprint 8

```bash
streamlit run streamlit_app.py
```


Modern ayri GUI katmani:

```bash
streamlit run gui/dashboard.py
```
### Sprint 9

```bash
python webcam_demo.py --transformation aging --intensity 0.6
```

```bash
python webcam_demo.py --transformation reference_expression_transfer --reference-image samples/test_face_2.png --intensity 0.75
```

### Sprint 10

```bash
python -c "from app import run_reference_expression_transfer_pipeline; r = run_reference_expression_transfer_pipeline('samples/test_face_1.jpg', 'samples/test_face_2.png', blend_factor=0.7, show_landmarks=True, selected_regions=['eyes','eyebrows','nose','lips']); print(r['transformation']['operation']); print(r['metrics']['mse']); print(r['metrics']['psnr']); print(r['transformation']['warped_image_path'])"
```

## Test Verileri

Repo icinde test icin uc ornek yuz vardir:

- `samples/test_face_1.jpg`
- `samples/test_face_2.png`
- `samples/test_face_3.jpg`

Bu gorseller GUI olmadan, komut satiri uzerinden preprocessing, face detection, landmark detection, warping, frequency analysis, aging/de-aging, webcam ve reference-transfer katmanlarini tek tek dogrulamak icin kullanilir.

## Muhendislik Ozeti

Bu noktaya kadar proje, "ham fotografi alip yuzu degistiren bir uygulama" olmaktan cok, yuz donusum sisteminin **guvenilir veri hazirlama altyapisini** kurmus durumda.

Su an sistemin en onemli teknik kazanimlari:

- giris verisini standardize etme
- yuz ROI'sini ayirma
- yuzu landmark tabanli geometrik temsil haline getirme
- triangle mesh tabanli warping altyapisi kurma
- spektral enerji analizi yapma
- frequency-based aging/de-aging filtreleri kurma
- MSE / PSNR / SSIM degerlendirme modulu ekleme
- Streamlit tabanli interaktif arayuz kurma
- ilerideki DSP asamalari icin saglam veri hatti kurma

Bir sonraki dogal asama:

- GAN/diffusion tabanli generative expression transfer
- browser-icinde tam akici webcam streaming
- production-grade model secimi ve hiz optimizasyonu


