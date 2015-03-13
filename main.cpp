#include <iostream>
#include <fstream>

#include "DecisionTree.h"
#include "RandomForest.h"

template<class T> void printVector(const std::vector<T> &vec)
{
    for(auto &x : vec)
        std::cout << x << " ";
    std::cout << std::endl;
}

template<class T> void splitAt(
        const std::vector<T> &lines, int half_size, 
        std::vector<T> &split_lo, std::vector<T> &split_hi)
{
    split_lo = std::vector<T>(lines.begin(), lines.begin() + half_size);
    split_hi = std::vector<T>(lines.begin() + half_size, lines.end());
}

void readGlass(ml::DataSet &dataset, std::vector<int> &labels)
{
    std::ifstream in;
    in.open("glass.test");
    int ind;
    float val;
    int counter = 0;
    
    while(!in.eof()){
        in >> ind;
        ml::FeatureVec *row = new ml::FeatureVec(9);
        for(int i = 0; i < 9; ++i){
            in >> val;
            (*row)[i] = val;
        }
        dataset.push_back(row);
        in >> ind;
        labels.push_back(ind);
        counter++;
        if(counter > 213)
            break;
    }
    in.close();
}

void glassTest()
{
    ml::DataSet allData, d1, d2;
    std::vector<int> allLabels, l1, l2;
    readGlass(allData, allLabels);
    splitAt(allData,   150, d1, d2);
    splitAt(allLabels, 150, l1, l2);

    printf("before train\n");
    //ml::NodeBase *tree = ml::buildNode(d1, l1, 8);
    //ml::NodeBase *tree = ml::buildNodeWithRandom(d1, l1, 8);
    ml::RandomForest *tree = new ml::RandomForest(-1, 0.8, 15);//(d1, l1, 8);
    tree->fit(d1, l1);
    printf("after train\n");
    int err = 0;
    for(int i = 0; i < d2.size(); ++i){
        int ans = ml::bestFreq(tree->predict(*d2[i]));
        //int ans = ml::bestFreq(forest.predict(*d2[i]));
        if(ans != l2[i])
            err++;
        ////printf("ans %d %d\n", ans, l2[i]);
    }
    printf("errors %f %d %zu\n", (float)err/d2.size(), err, d2.size());
    //printf("ans %d\n", ml::bestFreq(tree->predict({
    //                1.51793, 12.79, 3.50, 1.12, 73.03, 0.64, 8.77, 0.00, 0.00})));
    //printf("ans %d\n", ml::bestFreq(tree->predict({
    //                1.51793, 12.79, 3.50, 1.12, 73.03, 0.64, 8.77, 0.00, 0.00})));
}

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

ml::DataSet readMNISTData(std::string fname)
{
    ml::DataSet res;
    std::ifstream in;
    in.open(fname.c_str(), std::ios::binary | std::ios::in);

    int magic;
    in.read((char*)&magic, sizeof(magic));
    magic = reverseInt(magic);
    if(magic != 2051){
        throw std::string("bad magic");
    }
    int countOfImages;
    in.read((char*)&countOfImages, sizeof(countOfImages));
    countOfImages = reverseInt(countOfImages);

    int rows;
    in.read((char*)&rows, sizeof(rows));
    rows = reverseInt(rows);

    int cols;
    in.read((char*)&cols, sizeof(cols));
    cols = reverseInt(cols);

    printf("images %d rows %d cols %d\n", countOfImages, rows, cols);

    res.resize(countOfImages, NULL);
    for(int i = 0; i < countOfImages; ++i){
        ml::FeatureVec *feats = new ml::FeatureVec(rows*cols, 0);
        for(int r = 0; r < rows; ++r){
            for(int c = 0; c < cols; ++c){
                unsigned char temp=0;
                in.read((char*)&temp,sizeof(temp));
                (*feats)[r*rows + c] = temp;
                //printf("%f\n", (float)temp);
            }
        }
        res[i] = feats;
    }

    in.close();
    return res;
}

std::vector<int> readMNISTLabels(std::string fname)
{
    std::ifstream in;
    in.open(fname.c_str(), std::ios::binary | std::ios::in);
    int magic;
    in.read((char*)&magic, sizeof(magic));
    magic = reverseInt(magic);

    int countOfLabels;
    in.read((char*)&countOfLabels, sizeof(countOfLabels));
    countOfLabels = reverseInt(countOfLabels);

    printf("magic %d %d\n", magic, countOfLabels);

    std::vector<int> res(countOfLabels, 0);
    unsigned char tmp;
    int counter = 0;
    while(in.good()){
        in.read((char*)&tmp, sizeof(tmp));
        res[counter] = tmp;
        counter++;
    }

    in.close();
    return res;
}

void mnistTest()
{
    ml::DataSet data = readMNISTData("/home/daiver/Downloads/train-images-idx3-ubyte");
    std::vector<int> labels = readMNISTLabels("/home/daiver/Downloads/train-labels-idx1-ubyte");
    ml::DataSet dataT = readMNISTData("/home/daiver/Downloads/t10k-images-idx3-ubyte");
    std::vector<int> labelsT = readMNISTLabels("/home/daiver/Downloads/t10k-labels-idx1-ubyte");
    printf("before train\n");
    //ml::NodeBase *tree = ml::buildNode(d1, l1, 8);
    //ml::NodeBase *tree = ml::buildNodeWithRandom(d1, l1, 8);
    //ml::RandomForest *tree = new ml::RandomForest(0.04, 0.50, 200);//0.0323
    //ml::RandomForest *tree = new ml::RandomForest(0.03, 0.70, 200);//0.0314
    //ml::RandomForest *tree = new ml::RandomForest(0.035, 0.80, 250);//0.0303
    //ml::RandomForest *tree = new ml::RandomForest(0.035, 1.20, 250);//0.0286
    //ml::RandomForest *tree = new ml::RandomForest(0.075, 1.20, 250);//0.0277
    ml::RandomForest *tree = new ml::RandomForest(0.100, 1.20, 25);//
    tree->fit(data, labels);
    printf("after train\n");
    int err = 0;
    int testCount = dataT.size();

    for(int i = 0; i < testCount; ++i){
        int ans = ml::bestFreq(tree->predict(*dataT[i]));
        //int ans = ml::bestFreq(forest.predict(*d2[i]));
        if(ans != labelsT[i])
            err++;
        //printf("%d\n", ans);
    }
    printf("errors %f %d %d\n", (float)err/testCount, err, testCount);

}

int main()
{
    srand(42);
    //glassTest();
    mnistTest();
    return 0;
}
