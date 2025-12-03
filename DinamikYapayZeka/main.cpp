#include <iostream>
#include <random>

float Random01();


class Layer {
public:
	int mGirdiSayisi;
	int mNoronSayisi;
	float** mAgirliklar;
	float* mBiaslar;

	float* mNetGirdi;
	float* mNetCikti;
	float lr;
	Layer(int girdi, int noron) :mGirdiSayisi(girdi), mNoronSayisi(noron) {

		mNetGirdi = new float[mNoronSayisi];
		mNetCikti = new float[mNoronSayisi];
		lr = 0.001f;

		/*--Agirliklari olustur----*/
		mAgirliklar = new float* [mNoronSayisi];

		for (int i = 0; i < mNoronSayisi; i++)
		{
			mAgirliklar[i] = new float[mGirdiSayisi];
		}
		// Biaslari olustur..
		mBiaslar = new float[mNoronSayisi];
		//Agirliklara ve Biasa Rastgele Deger Ata & Deltalari sifirla
		for (int i = 0; i < mNoronSayisi; i++)
		{
			mBiaslar[i] = Random01();

			for (int j = 0; j < mGirdiSayisi; j++)
			{
				mAgirliklar[i][j] = Random01();
			}
		}

	}

	void KismiIleriBesleme(float* X) {
		// Girdi...
		for (int i = 0; i < mNoronSayisi; i++)
		{
			mNetGirdi[i] = 0.0f;
			for (int j = 0; j < mGirdiSayisi; j++)
			{
				mNetGirdi[i] += mAgirliklar[i][j] * X[j];
			}
			mNetGirdi[i] += mBiaslar[i];
		}
		//Cikti...
		for (int i = 0; i < mNoronSayisi; i++)
		{
			mNetCikti[i] = (mNetGirdi[i] < 0) ? 0 : mNetGirdi[i];  //RelU aktivasyon Fonksiyonu
		}

	}

	float RelUDer(float x) {
		return x < 0 ? 0.0f : 1.0f;
	}

	void AgirliklariGuncelle(float** dAgirlik, float* dBias) {
		for (int i = 0; i < mNoronSayisi; i++)
		{
			mBiaslar[i] -= lr * dBias[i];
			for (int j = 0; j < mGirdiSayisi; j++)
			{
				mAgirliklar[i][j] -= lr * dAgirlik[i][j];
			}
		}

	}
	~Layer() {

		delete[] mNetGirdi;
		delete[] mNetCikti;
		delete[] mBiaslar;


		for (int i = 0; i < mNoronSayisi; i++) {
			delete[] mAgirliklar[i];
		}
		delete[] mAgirliklar;
	}
};

class NeuralNetwork {
private:
	Layer* GizliKatman;
	Layer* CikisKatmani;
	float outputDelta;
	float* hiddenDelta;
	int mNoronSayisi;
	int mGirdiSayisi;
	float** Xler;
	float* Yler;
	int msatirSayisi;
	float* mHatalar;
	float** derAgirlikGizli;
	float** derAgirlikCikis;
	float* derBiasGizli;
	float* derBiasCikis;

public:

	NeuralNetwork(int Noronsayisi, int girdiSayisi, int satirSayisi) :mNoronSayisi(Noronsayisi), mGirdiSayisi(girdiSayisi), msatirSayisi(satirSayisi) {
		outputDelta = 0.0f;
		hiddenDelta = new float[mNoronSayisi];
		GizliKatman = new Layer(mGirdiSayisi, mNoronSayisi);
		CikisKatmani = new Layer(mNoronSayisi, 1);
		mHatalar = new float[msatirSayisi];
		Yler = nullptr;
		Xler = nullptr;

		derAgirlikGizli = new float* [mNoronSayisi];
		derAgirlikCikis = new float* [1];
		derAgirlikCikis[0] = new float[mNoronSayisi];
		derBiasGizli = new float[mNoronSayisi];
		derBiasCikis = new float[1];
		for (int i = 0; i < mNoronSayisi; i++)
		{
			derAgirlikGizli[i] = new float[mGirdiSayisi];
		}
		for (int i = 0; i < mNoronSayisi; i++)
		{
			hiddenDelta[i] = 0.0f;
		}
		for (int i = 0; i < msatirSayisi; i++)
		{
			mHatalar[i] = 0.0f;
		}



	}
	void ForwardPass(float* GirisVeri) {
		GizliKatman->KismiIleriBesleme(GirisVeri);
		CikisKatmani->KismiIleriBesleme(GizliKatman->mNetCikti);
	}

	void BackPropagation(float gercek, float tahmin, int satir) {
		// Deltalarý Hesapla...
		outputDelta = (tahmin - gercek) * (CikisKatmani->RelUDer(CikisKatmani->mNetGirdi[0]));
		for (int i = 0; i < mNoronSayisi; i++)
		{
			hiddenDelta[i] = outputDelta * CikisKatmani->mAgirliklar[0][i] * GizliKatman->RelUDer(GizliKatman->mNetGirdi[i]);

		}

		derBiasCikis[0] = outputDelta;
		for (int j = 0; j < mNoronSayisi; j++)
		{
			derAgirlikCikis[0][j] = outputDelta * (GizliKatman->mNetCikti[j]);
		}

		for (int i = 0; i < mNoronSayisi; i++)
		{
			derBiasGizli[i] = hiddenDelta[i];
			for (int j = 0; j < mGirdiSayisi; j++)
			{
				derAgirlikGizli[i][j] = hiddenDelta[i] * Xler[satir][j];
			}
		}
	}

	void Egitim(int epoch) {
		for (int e = 0; e < epoch; e++)
		{
			float toplamHata = 0.0f;

			for (int i = 0; i < msatirSayisi; i++)
			{
				ForwardPass(Xler[i]);

				float hata = Yler[i] - CikisKatmani->mNetCikti[0];
				toplamHata += hata * hata;

				BackPropagation(Yler[i], CikisKatmani->mNetCikti[0], i);
				GizliKatman->AgirliklariGuncelle(derAgirlikGizli, derBiasGizli);
				CikisKatmani->AgirliklariGuncelle(derAgirlikCikis, derBiasCikis);
			}

			if (e % 100 == 0) {
				std::cout << "Epoch: " << e << " | Ortalama Hata: " << (toplamHata / msatirSayisi) << std::endl;
			}
		}

	}

	void VeriSetiGir(float** X, float* Y) {
		Xler = X;
		Yler = Y;
	}

	float TahminEt(float* veri) {
		ForwardPass(veri);
		return CikisKatmani->mNetCikti[0];
	}

	~NeuralNetwork() {
		delete GizliKatman;
		delete CikisKatmani;
		delete[] hiddenDelta;
		delete[] mHatalar;


		for (int i = 0; i < mNoronSayisi; i++) delete[] derAgirlikGizli[i];
		delete[] derAgirlikGizli;

		delete[] derAgirlikCikis[0];
		delete[] derAgirlikCikis;

		delete[] derBiasGizli;
		delete[] derBiasCikis;
	}



};

int main() {


	int satirSayisi = 4;
	int girdiSayisi = 2;
	int gizliNoronSayisi = 50;



	float** X = new float* [satirSayisi];
	float* Y = new float[satirSayisi];


	X[0] = new float[2];
	X[0][0] = 0.0f; X[0][1] = 0.0f;
	Y[0] = 0.0f;


	X[1] = new float[2];
	X[1][0] = 0.0f; X[1][1] = 1.0f;
	Y[1] = 1.0f;


	X[2] = new float[2];
	X[2][0] = 1.0f; X[2][1] = 0.0f;
	Y[2] = 1.0f;


	X[3] = new float[2];
	X[3][0] = 1.0f; X[3][1] = 1.0f;
	Y[3] = 0.0f;


	NeuralNetwork ysa(gizliNoronSayisi, girdiSayisi, satirSayisi);
	ysa.VeriSetiGir(X, Y);


	ysa.Egitim(20000);

	std::cout << "\n--- TEST SONUCLARI ---\n";

	float test1[] = { 0,0 };
	float test2[] = { 1,1 };
	float test3[] = { 0,1 };
	float test4[] = { 1,0 };

	std::cout << "0,0----->" << ysa.TahminEt(test1) << std::endl;
	std::cout << "1,1----->" << ysa.TahminEt(test2) << std::endl;
	std::cout << "0,1----->" << ysa.TahminEt(test3) << std::endl;
	std::cout << "1,0----->" << ysa.TahminEt(test4) << std::endl;

	delete[] Y;
	for (int i = 0; i < satirSayisi; i++) delete[] X[i];
	delete[] X;

	return 0;
}

float Random01() {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_real_distribution<> dist(0.0, 1.0);
	return dist(gen);
}

