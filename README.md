# E-Ticaret Ürün Yorumları Sentiment Analizi

Bu proje, e-ticaret platformlarında yapılan ürün yorumlarını analiz ederek duygu sınıflandırması yapan bir **LSTM (Long Short-Term Memory)** modeli geliştirmektedir.

## İçindekiler
- [Genel Bakış](#genel-bakış)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Mimarisi](#model-mimarisi)
- [Eğitim Süreci](#eğitim-süreci)
- [Tahmin Fonksiyonu](#tahmin-fonksiyonu)
- [Sonuçlar](#sonuçlar)

## Genel Bakış
Bu proje, e-ticaret sitelerindeki müşteri yorumlarını **olumlu, olumsuz veya nötr** olarak sınıflandırmak amacıyla **derin öğrenme tabanlı** bir model kullanmaktadır. **Doğal dil işleme (NLP)** teknikleri ile metin verisi işlenerek analiz edilmektedir.

## Gereksinimler
Aşağıdaki Python kütüphanelerini yüklemeniz gerekmektedir:

```bash
pip install pandas numpy tensorflow scikit-learn
```

## Kurulum
1. **Gereksinimleri yükleyin**.
2. **Veri dosyasını ekleyin**: `e-ticaret_urun_yorumlari.csv` dosyasını proje dizinine ekleyin.
3. **Notebook dosyasını çalıştırın**:

```bash
jupyter notebook e_ticaret_urun_yorumlari.ipynb
```

## Kullanım
- `e-ticaret_urun_yorumlari.csv` dosyasındaki yorumlar okunur ve temizlenir.
- Tokenizer ile metinler sayısal değerlere dönüştürülür.
- LSTM modeli eğitilir ve test edilir.
- Kullanıcı girdileri ile modelin tahmin yapması sağlanır.

## Model Mimarisi
Model, aşağıdaki katmanlardan oluşmaktadır:
- **Embedding Layer**: Kelimeleri vektörlere dönüştürme.
- **3 Adet LSTM Katmanı**: Sıralı verileri işlemek için.
- **Dropout Katmanları**: Overfitting’i önlemek için.
- **Dense Katmanı**: Çıktıyı sınıflandırmak için.

```python
model = Sequential()
model.add(Embedding(input_dim=4000, output_dim=100, input_length=40))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
```

## Eğitim Süreci
- **Veriler eğitim ve test setlerine ayrılmıştır.**
- **Categorical crossentropy** kayıp fonksiyonu kullanılmıştır.
- **Adam optimizer** ile model derlenmiştir.
- **EarlyStopping** ile en iyi ağırlıklar geri yüklenmektedir.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Tahmin Fonksiyonu
Aşağıdaki fonksiyon, girilen metin üzerinden duygu tahmini yapmaktadır.

```python
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)
    sentiment = np.argmax(prediction)
    return sentiment  # 0: olumsuz, 1: olumlu, 2: nötr
```

## Sonuçlar
Eğitim sonunda elde edilen **doğruluk oranı**:
- **Eğitim Seti Doğruluğu**: %98.0
- **Test Seti Doğruluğu**: %88.2

Örnek tahmin:
```python
predict_sentiment("Mükemmel bir ürün!")  # Çıktı: Olumlu
```
