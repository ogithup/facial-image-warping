# Facial Image Warping

Bu repo, **Facial Image Warping, Aging, and Expression Transformation** projesinin ilk üç sprintinin çalışan temelini içerir. Şu ana kadar odak noktası, yüzü değiştirmekten önce veriyi **doğru almak**, **DSP açısından standardize etmek**, **yüz bölgesini ayırmak** ve **yüzü landmark noktalarıyla temsil etmek** oldu.

Mevcut durumda tamamlanan katmanlar:

1. `Sprint 1` — Image input ve preprocessing
2. `Sprint 2` — Face detection ve face crop
3. `Sprint 3` — Facial landmark detection ve landmark export

Henüz tamamlanmayan katmanlar:

- Geometric warping
- Aging / de-aging
- Fourier analysis
- Quantitative evaluation

## DSP Mantığı

Bu projede görüntü, klasik bir fotoğraf dosyası gibi değil, **2 boyutlu sayısal sinyal** olarak ele alınır. Kurulan akış şu mantığa dayanır:

`User Input -> Image Acquisition -> Preprocessing -> ROI Extraction -> Landmark Representation -> Future DSP / Warping Stages`

Bu zincirde şu ana kadar yapılan işler:

- **Image acquisition**: giriş dosyasını doğrulama ve belleğe alma
- **Preprocessing**: renk uzayı dönüşümü, grayscale, resize, normalize etme
- **ROI extraction**: tüm görüntü yerine yalnızca yüz bölgesini seçme
- **Geometric representation**: yüzü 468 landmark ile parametrik olarak ifade etme

Bu sayede ileride yapılacak warping, aging ve frekans analizi işlemleri doğrudan ham görüntüye değil, temizlenmiş ve anlamlı hale getirilmiş yüz verisine uygulanır.

## Şu Ana Kadarki Mimari

### Ana modüller

1. `input_module.py`
   Giriş dosyasını doğrular, görüntüyü yükler, metadata üretir.
2. `preprocessing.py`
   `BGR/RGB/grayscale`, resize, normalization, histogram ve processed image üretir.
3. `face_detection.py`
   OpenCV Haar Cascade ile yüz bulur, bounding box çizer, yüz crop alır.
4. `landmark_detection.py`
   MediaPipe Face Mesh ile landmark çıkarır, JSON/CSV export yapar.
5. `app.py`
   Sprint bazlı pipeline giriş noktalarını sağlar.

### Pipeline giriş noktaları

- `run_preprocessing_pipeline(...)`
- `run_face_detection_pipeline(...)`
- `run_landmark_pipeline(...)`

## Örnek Akış

Aşağıdaki örnek, `samples/test_face_1.jpg` üzerinden sistemin şu ana kadar neleri ürettiğini gösterir.

### 1. Örnek giriş yüzü

Bu görsel sistemin aldığı ham inputtur. Bu aşamada henüz sinyal seçimi yoktur; arka plan ve yüz birlikte işlenir.

![Sample Face](docs/images/sample-face.jpg)

### 2. Preprocessed face

Bu çıktı preprocessing adımından gelir:

- görüntü okunur
- standart çözünürlüğe taşınır
- normalize edilir
- sonraki aşamaların tutarlı çalışması için standardize edilir

Bu adımın DSP karşılığı, giriş sinyalini ortak bir ölçeğe çekmek ve sonraki işlemlere uygun hale getirmektir.

![Preprocessed Face](docs/images/preprocessed-face.png)

### 3. Histogram çıktısı

Histogram, görüntü yoğunluk dağılımını gösterir. Bu, sinyalin enerji dağılımı kadar güçlü bir analiz değildir ama preprocessing aşamasında parlaklık ve kontrast karakterini anlamak için temel bir araçtır.

Bu çıktı özellikle şu sorular için anlamlıdır:

- görüntü çok karanlık mı
- çok açık mı
- kontrast dar mı geniş mi

![Histogram](docs/images/histogram.png)

### 4. Detected face

Bu aşamada OpenCV Haar Cascade ile yüz bulunur ve bounding box çizilir.

Burada yapılan iş:

- nesne tespiti
- koordinat üretimi
- ROI seçimi için yüz sınırını tanımlama

DSP açısından bu adım, bütün sinyalden sadece anlamlı alt bölgeyi ayırmaya giden ilk adımdır.

![Detected Face](docs/images/detected-face.png)

### 5. Face crop

Bounding box kullanılarak yüz bölgesi kesilir ve ayrı bir görüntü olarak normalize edilir.

Bu adım çok kritiktir çünkü sonraki landmark, warping ve aging işlemleri tüm fotoğraf üzerinde değil, yalnızca bu ROI üzerinde yapılacaktır.

Burada öğrendiğimiz kavramlar:

- `bounding box`
- `ROI`
- `crop`
- `coordinate system`

![Face Crop](docs/images/face-crop.png)

### 6. Landmark çıktısı

Sprint 3 ile MediaPipe Face Mesh üzerinden landmark üretimi eklenmiştir. Çalışan akış sonunda şu dosyalar üretilir:

- `outputs/landmarks/<name>_landmarks.png`
- `outputs/landmarks/<name>_landmarks.json`
- `outputs/landmarks/<name>_landmarks.csv`

Bu landmark katmanında:

- yüz 468 veya refine-landmarks ile daha yüksek sayıda landmark noktasıyla temsil edilir
- seçili bölgeler (`eyes`, `nose`, `lips`, `eyebrows`, `jawline`, `cheeks`) ayrıca işaretlenebilir
- koordinatlar hem normalize hem piksel uzayında dışa aktarılır

Aşağıdaki örnek görsel, `test_face_1.jpg` için üretilen gerçek landmark overlay çıktısıdır:

![Landmark Face](docs/images/landmarks-face.png)

Bu aşama, geometric warping için kritik veri üretir. Sonraki sprintlerde ağız, kaş, göz, çene ve yanak deformasyonları doğrudan bu landmark koordinatları üzerinden yapılacaktır. Yani bu görsel yalnızca bir overlay değildir; yüzün parametrik temsilidir.

## Landmark Koordinatlarının Mantığı

MediaPipe, landmark noktalarını önce **normalize koordinat** olarak verir:

- `x` değeri genişliğe göre `0-1`
- `y` değeri yüksekliğe göre `0-1`

Bu yaklaşımın avantajı:

- farklı çözünürlüklerde aynı yüz geometrisini taşımak kolaylaşır
- model ekran pikseline değil göreli konuma konuşur

Ama OpenCV çizimi ve crop işlemleri için bu koordinatlar piksele çevrilmelidir. Bu yüzden `landmark_detection.py` içinde normalize değerler mutlak piksele dönüştürülür.

## Şu Ana Kadar Hangi Gereksinimler Tamamlandı

### Sprint 1

- JPG ve PNG yükleme
- dosya uzantısı ve varlık doğrulama
- `BGR -> RGB -> GRAYSCALE` dönüşümleri
- standart çözünürlükte resize
- piksel normalizasyonu
- histogram oluşturma
- processed image kaydetme

### Sprint 2

- frontal face detection
- bounding box çizme
- detected face preview kaydetme
- crop alma
- crop’u normalize etme
- yüz bulunamazsa açık hata verme

### Sprint 3

- MediaPipe ile face mesh çıkarma
- full landmark visualization
- region-based visualization
- JSON export
- CSV export
- landmark visualization toggle

## Dosya Yapısı

```text
facial-image-warping/
├── app.py
├── pyproject.toml
├── README.md
├── docs/
│   └── images/
├── samples/
├── src/
│   └── facial_image_warping/
│       ├── face_detection.py
│       ├── input_module.py
│       ├── landmark_detection.py
│       ├── preprocessing.py
│       └── ...
└── tests/
```

## Kurulum

Bu proje için sürümler, “en yeni olan” mantığıyla değil, **koddaki API kullanımıyla uyumlu** olacak şekilde seçildi.

Özellikle:

- `opencv-python>=4.10,<5`
- `mediapipe==0.10.9`
- `numpy>=1.26,<2.0`

Kurulum:

```bash
pip install -e ".[dev]"
```

## Çalıştırma

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

## Test Verileri

Repo içinde test için üç örnek yüz vardır:

- `samples/test_face_1.jpg`
- `samples/test_face_2.png`
- `samples/test_face_3.jpg`

Bu görseller GUI olmadan, komut satırı üzerinden preprocessing, face detection ve landmark detection katmanlarını tek tek doğrulamak için kullanılır.

## Mühendislik Özeti

Bu noktaya kadar proje, “ham fotoğrafı alıp yüzü değiştiren bir uygulama” olmaktan çok, yüz dönüşüm sisteminin **güvenilir veri hazırlama altyapısını** kurmuş durumda.

Şu an sistemin en önemli teknik kazanımları:

- giriş verisini standardize etme
- yüz ROI’sini ayırma
- yüzü landmark tabanlı geometrik temsil haline getirme
- ilerideki warping ve DSP aşamaları için sağlam veri hattı kurma

Bir sonraki doğal aşama:

- `geometric_warping.py` içinde landmark tabanlı yüz deformasyonu
- ardından aging/de-aging ve Fourier tabanlı analiz katmanları
