#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <stdlib.h>
#include <algorithm>
#include <string.h>

#include "common.h"
#include "Node.h"

namespace ml {
NodeBase *buildNode(
        const DataSet &dataset, const std::vector<int> &labels, int countOfClasses);

NodeBase *buildNodeWithRandom(
        const DataSet &dataset, const std::vector<int> &labels, int countOfClasses, float featureSamplingCoeff);

std::vector<float> freqsFromLabels(const std::vector<int> &labels, int countOfClasses);

void findbestThreshold(
    const DataSet &dataset, const std::vector<int> &labels, int countOfClasses,
    const std::vector<int> &features,
    DataSet *d1, DataSet *d2, std::vector<int> *l1, std::vector<int> *l2,
    int *bestIndex, float *bestThreshold, float *bestGain);

void findbestThresholdFast(
    const DataSet &dataset, const std::vector<int> &labels, int countOfClasses,
    const std::vector<int> &features,
    int *bestIndex, float *bestThreshold, float *bestGain);


void divide(
        const DataSet &dataset, const std::vector<int> &labels,
        int featureIndex, float featureThreshold,
        DataSet *d1, DataSet *d2, std::vector<int> *l1, std::vector<int> *l2);

float gini(const std::vector<int> &labels, int countOfClasses);

}

#endif
