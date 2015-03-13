#include "RandomForest.h"

using namespace ml;

RandomForest::RandomForest(
        float featureSamplingCoeff, 
        float dataSamplingCoeff, 
        int countOfTrees): 
    featureSamplingCoeff(featureSamplingCoeff), dataSamplingCoeff(dataSamplingCoeff)
{
    this->countOfClasses = 0;
    this->ansamble.clear();
    this->ansamble.resize(countOfTrees, NULL);
}

std::vector<float> ml::RandomForest::predict(const FeatureVec &x)
{
    if(ansamble[0] == 0){
        throw std::string("Forest are not initialized\n");
    }
    std::vector<float> res(this->countOfClasses, 0);
    for(int i = 0; i < this->ansamble.size(); ++i){
        std::vector<float> ans = this->ansamble[i]->predict(x);
        for(int j = 0; j < ans.size(); ++j)
            res[j] += ans[j];
    }
    for(int j = 0; j < res.size(); ++j)
        res[j] /= this->ansamble.size();
    return res;
}

void ml::RandomForest::fit(const DataSet &dataset, const std::vector<int> &labels)
{

    int maxClass = 0;

    for(int i = 0; i < labels.size(); ++i)
        if(labels[i] > maxClass)
            maxClass = labels[i];
    this->countOfClasses = maxClass + 1;

    if(this->featureSamplingCoeff < 0)
        this->featureSamplingCoeff = sqrt((float)dataset[0]->size())/dataset[0]->size();

    const int targetCountOfData = dataset.size() * dataSamplingCoeff;
    DataSet d(targetCountOfData);
    std::vector<int> l(targetCountOfData);
    for(int ansambleInd = 0; ansambleInd < ansamble.size(); ++ansambleInd){
        for(int i = 0; i < targetCountOfData; ++i){
            int ind = rand() % dataset.size();
            d[i] = dataset[ind];
            l[i] = labels[ind];
        }
        ansamble[ansambleInd] = buildNodeWithRandom(d, l, maxClass + 1, this->featureSamplingCoeff);
        printf("Tree %d trained\n", ansambleInd);
    }
}

