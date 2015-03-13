#include "Node.h"

using namespace ml;

NodeBranch::NodeBranch(
        int featureIndex, 
        float featureThreshold,
        NodeBase *left,
        NodeBase *right): 
    featureIndex(featureIndex), featureThreshold(featureThreshold), 
    left(left), right(right)
{
}

std::vector<float> NodeBranch::predict(const FeatureVec &x)
{
    if(x[this->featureIndex] >= this->featureThreshold)
        return this->right->predict(x);
    return this->left->predict(x);
}

NodeLeaf::NodeLeaf(const std::vector<float> &freqs): freqs(freqs)
{
    
}

std::vector<float> NodeLeaf::predict(const FeatureVec &)
{
    return this->freqs;
}
