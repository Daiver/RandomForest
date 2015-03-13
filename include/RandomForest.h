#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <vector>
#include <string>

#include "Node.h"
#include "DecisionTree.h"
#include "common.h"

namespace ml {

class RandomForest{
public:
    RandomForest(float featureSamplingCoeff, float dataSamplingCoeff, int countOfTrees);

    std::vector<float> predict(const FeatureVec &x);
    void fit(const DataSet &dataset, const std::vector<int> &labels);

protected:
    std::vector<NodeBase *> ansamble;

    float featureSamplingCoeff;
    float dataSamplingCoeff;
    int countOfClasses;
};

}

#endif
