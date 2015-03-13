#ifndef NODE_H
#define NODE_H

#include <vector>

#include "common.h"

namespace ml {

class NodeBase
{
public:
    virtual std::vector<float> predict(const FeatureVec &x) = 0;
};

class NodeBranch: public NodeBase
{
public:
    NodeBranch(
            int featureIndex, 
            float featureThreshold,
            NodeBase *left,
            NodeBase *right);

    std::vector<float> predict(const FeatureVec &x);

protected:
    int featureIndex;
    float featureThreshold;
    NodeBase *left;
    NodeBase *right;

};

class NodeLeaf: public NodeBase
{
public:
    NodeLeaf(const std::vector<float> &freqs);
    std::vector<float> predict(const FeatureVec &x);

protected:
    std::vector<float> freqs;
};

}

#endif
