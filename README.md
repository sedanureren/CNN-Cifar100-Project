# CNN-Cifar100-Project
Bu proje, CIFAR-100 veri seti kullanılarak çok-sınıflı sınıflandırma modeli eğitmek amacıyla gerçekleştirilmiştir. Proje kapsamında öğrenci numarama göre belirlenen 7 sınıf seçilerek bu sınıflar üzerinde bir konvolüsyonel sinir ağı (CNN) modeli oluşturulmuş ve eğitilmiştir.

Projenin adımları şu şekildedir:

Veri Hazırlığı ve Görselleştirme: CIFAR-100 veri setinden belirlenen 7 sınıf için örnek görüntüler seçilmiş ve gösterilmiştir.
Modelin Tasarımı: Model, 3 veya 4 adet konvolüsyon katmanı ve en az iki adet Dense katmanı içerecek şekilde tasarlanmıştır. Ek olarak MaxPooling, AvgPooling ve Dropout katmanları da modele dahil edilmiştir. Modelin blok şeması çizilmiştir.
Modelin Eğitimi: Model uygun epoch sayısı ile eğitilmiş ve doğruluk (accuracy) ile kayıp (loss) grafiklerini içerecek şekilde görselleştirilmiştir. Eğitim ve doğrulama (validation) verileri kullanılmış, bu grafikler yorumlanmıştır.
Modelin Test Edilmesi: Eğitilen model, test veri seti kullanılarak test edilmiş ve karmaşıklık (confusion) matrisi elde edilmiştir.
Tahminlerin Gösterilmesi: Eğitilen model, predict fonksiyonu kullanılarak her sınıftan örnek bir görüntü için test edilmiş ve elde edilen çıkış vektörleri gösterilmiştir.
Bu proje, derin öğrenme ve görüntü sınıflandırma konularında pratik deneyim kazanmak amacıyla gerçekleştirilmiştir.
