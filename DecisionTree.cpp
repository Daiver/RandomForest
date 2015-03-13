#include "DecisionTree.h"

using namespace ml;

NodeBase *ml::buildNodeWithRandom(
        const DataSet &dataset, const std::vector<int> &labels, 
        int countOfClasses, float featureSamplingCoeff)
{
    int bestIndex; float bestThreshold; float bestGain;
    DataSet d1, d2; 
    std::vector<int> l1, l2;


    const int countOfFeatures = dataset[0]->size();

    const int targetCountOfFeatures = countOfFeatures*featureSamplingCoeff;

    std::vector<int> features(countOfFeatures);
    for(int i = 0; i < countOfFeatures; ++i)
        features[i] = i;
    ml::shuffle(&features);
    std::vector<int> features2(targetCountOfFeatures);
    for(int i = 0; i < targetCountOfFeatures; ++i){
        features2[i] = features[i];
    }

    findbestThresholdFast(
            dataset, labels, countOfClasses, features2,
            &bestIndex, &bestThreshold, &bestGain);
    //printf("best gain %f %d\n", bestGain, labels.size());
    const float minGain = 0.0000000001;
    if(bestGain > minGain )
        divide(dataset, labels, bestIndex, bestThreshold, &d1, &d2, &l1, &l2);
    if(bestGain < minGain || l1.size() == 0 || l2.size() == 0){
        std::vector<float> freqs = freqsFromLabels(labels, countOfClasses);
        return new NodeLeaf(freqs);
    }
    NodeBase *left  = buildNodeWithRandom(d1, l1, countOfClasses, featureSamplingCoeff);
    NodeBase *right = buildNodeWithRandom(d2, l2, countOfClasses, featureSamplingCoeff);
    return new NodeBranch(bestIndex, bestThreshold, left, right);
}

NodeBase *ml::buildNode(
        const DataSet &dataset, const std::vector<int> &labels, int countOfClasses)
{
    int bestIndex; float bestThreshold; float bestGain;
    DataSet d1, d2; 
    std::vector<int> l1, l2;

    const int countOfFeatures = dataset[0]->size();
    std::vector<int> features(countOfFeatures);
    for(int i = 0; i < countOfFeatures; ++i)
        features[i] = i;

    findbestThreshold(
            dataset, labels, countOfClasses, features,
            &d1, &d2, &l1, &l2, 
            &bestIndex, &bestThreshold, &bestGain);
    
    const float minGain = 0.01;
    if(bestGain < minGain || l1.size() == 0 || l2.size() == 0){
        std::vector<float> freqs = freqsFromLabels(labels, countOfClasses);
        return new NodeLeaf(freqs);
    }
    NodeBase *left  = buildNode(d1, l1, countOfClasses);
    NodeBase *right = buildNode(d2, l2, countOfClasses);
    return new NodeBranch(bestIndex, bestThreshold, left, right);
}

std::vector<float> ml::freqsFromLabels(const std::vector<int> &labels, int countOfClasses)
{
    std::vector<float> res(countOfClasses, 0);
    for(int i = 0; i < labels.size(); ++i)
        ++res[labels[i]];
    for(int i = 0; i < res.size(); ++i)
        res[i] /= labels.size();
    return res;
}

void ml::findbestThreshold(
    const DataSet &dataset, const std::vector<int> &labels, int countOfClasses,
    const std::vector<int> &features,
    DataSet *d1, DataSet *d2, std::vector<int> *l1, std::vector<int> *l2,
    int *bestIndex, float *bestThreshold, float *bestGain)
{

    *bestIndex     = 0;
    *bestGain      = 0;
    *bestThreshold = 0;
    float initGain = gini(labels, countOfClasses);
    printf("initGain %f\n", initGain);
    for(int i = 0; i < features.size(); ++i){
        for(int j = 0; j < dataset.size(); ++j){
            DataSet d1, d2; 
            std::vector<int> l1, l2;
            int f = features[i];
            divide(dataset, labels, f, (*dataset[j])[f], &d1, &d2, &l1, &l2);
            float p = (float)l1.size()/labels.size();
            float gain = initGain - p*gini(l1, countOfClasses) - (1 - p)*gini(l2, countOfClasses);
            if(*bestGain < gain){
                *bestGain = gain;
                *bestIndex = f;
                *bestThreshold = (*dataset[j])[f];
            }
        }
    }

    divide(dataset, labels, *bestIndex, *bestThreshold, d1, d2, l1, l2);
}

void ml::findbestThresholdFast(
    const DataSet &dataset, const std::vector<int> &labels, int countOfClasses,
    const std::vector<int> &features,
    int *bestIndex, float *bestThreshold, float *bestGain)
{
    if(labels.size() != dataset.size())
        printf("ERRRRR\n");
    float *hist  = new float[countOfClasses];
    float *histL = new float[countOfClasses];
    float *histR = new float[countOfClasses];

    memset(hist , 0, sizeof(float) * countOfClasses);

    for(int i = 0; i < labels.size(); ++i)
        ++hist[labels[i]];
    
    double g = 0, gl = 0, gr = 0;

    for(int i = 0; i < countOfClasses; ++i)
        g += hist[i]*hist[i];

    double inverseCountOfLabels = 1.0/labels.size();
    double initGain = 1.0 - g*inverseCountOfLabels*inverseCountOfLabels;
    //printf("initGain %f %f \n", initGain, inverseCountOfLabels*inverseCountOfLabels);

    struct SortFunctor{
        SortFunctor(int feature, const DataSet *dataset):
            feature(feature), dataset(dataset){}
        bool operator()(int i, int j) const
        {
            return (*(*dataset)[i])[feature] < (*(*dataset)[j])[feature];
        }
        const DataSet *dataset;
        int feature;
    };

    int *order = new int[labels.size()];

    *bestIndex = -1;
    for(int i = 0; i < labels.size(); ++i)
        order[i] = i;
    double bestG = initGain;
    int bestWl = 0, bestWr = 0;
    for(int featureIndex = 0; featureIndex < features.size(); ++featureIndex){
        int feature = features[featureIndex];
        gr = g;
        gl = 0;
        memcpy(histR, hist, sizeof(float) * countOfClasses);
        memset(histL, 0, sizeof(float) * countOfClasses);

        std::sort(order, order + labels.size(), SortFunctor(feature, &dataset));

        for(int j = 0; j < labels.size() - 1; ++j){
            int j1 = order[j];
            int j2 = order[j + 1];
            int label = labels[j1];

            gl -= histL[label] * histL[label];
            ++histL[label];
            gl += histL[label] * histL[label];

            gr -= histR[label] * histR[label];
            --histR[label];
            gr += histR[label] * histR[label];

            float wl = j + 1;
            float wr = labels.size() - wl;
            double curGain = (wl - gl/(wl)) * inverseCountOfLabels 
                           + (wr - gr/(wr)) * inverseCountOfLabels;
            float d1 = (*dataset[j1])[feature];
            float d2 = (*dataset[j2])[feature];
            
            if(curGain < bestG && d2 - d1 > 1e-6f){
                bestG = curGain;
                *bestIndex = feature;
                *bestThreshold = 0.5f * (d1 + d2);
                bestWl = wl;
                bestWr = wr;
            }
        }
    }

    delete[] order;
    delete[] hist;
    delete[] histL;
    delete[] histR;
    *bestGain = initGain - bestG;
}

float ml::gini(const std::vector<int> &labels, int countOfClasses)
{
    std::vector<float> freqs = freqsFromLabels(labels, countOfClasses);
    float sum = 1.0;
    for(int i = 0; i < freqs.size(); ++i)
        sum -= freqs[i] * freqs[i];
    return sum;
}

void ml::divide(
        const DataSet &dataset, const std::vector<int> &labels,
        int featureIndex, float featureThreshold,
        DataSet *d1, DataSet *d2, std::vector<int> *l1, std::vector<int> *l2)
{
    if(featureIndex == -1 )
        printf("bad featureIndex\n");
    for(int i = 0; i < dataset.size(); ++i){
        if((*dataset[i])[featureIndex] >= featureThreshold){
            d2->push_back(dataset[i]);
            l2->push_back(labels[i]);
        }else{
            d1->push_back(dataset[i]);
            l1->push_back(labels[i]);
        }
    }
}

