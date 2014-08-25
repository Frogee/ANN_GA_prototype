#include <cassert>
#include "neuralNetwork.h"

//////////////////////////////////////////////////////////////////////////
//NodeParameters_Omega

NodeParameters_Omega::NodeParameters_Omega() {
    //std::cout << "Empty NodeParameter_Omega constructor called" << std::endl;
}

NodeParameters_Omega::NodeParameters_Omega(std::string inputNodeName,
                                           std::vector<double> inputWeights_w,
                                           double inputBasalExpression_beta,
                                           double inputDecayRate_gamma,
                                           double inputScaleFactor_tau)
                                           : nodeName(inputNodeName),
                                           v_weights_w(inputWeights_w),
                                           basalExpression_beta(inputBasalExpression_beta),
                                           decayRate_gamma(inputDecayRate_gamma),
                                           scaleFactor_tau(inputScaleFactor_tau) {
    //std::cout << "NodeParameter_Omega initialized with values" << std::endl;
}

NodeParameters_Omega::~NodeParameters_Omega() {
}

void NodeParameters_Omega::SetNodeParameters(std::string inputNodeName,
                                             std::vector<double> inputWeights_w,
                                             double inputBasalExpression_beta,
                                             double inputDecayRate_gamma,
                                             double inputScaleFactor_tau) {
    this->nodeName = inputNodeName;
    this->v_weights_w = inputWeights_w;
    this->basalExpression_beta = inputBasalExpression_beta;
    this->decayRate_gamma = inputDecayRate_gamma;
    this->scaleFactor_tau = inputScaleFactor_tau;
}

void NodeParameters_Omega::PrintNodeParameters() {
    std::cout << "NodeParameters for " << nodeName << ":" << std::endl;
    std::cout << "Weights:\t";
    for (int i = 0; i < this->v_weights_w.size(); i++) {
        std::cout << this->v_weights_w[i] << "\t";
    }
    std::cout << std::endl;
    std::cout << "Basal Expression (beta):\t" << this->basalExpression_beta << std::endl;
    std::cout << "Decay Rate (gamma):\t" << this->decayRate_gamma << std::endl;
    std::cout << "Scale Factor (tau):\t" << this->scaleFactor_tau << std::endl;
}

std::string NodeParameters_Omega::GetNodeName() {
    return this->nodeName;
}

int NodeParameters_Omega::GetNumberWeights() {
    return this->v_weights_w.size();
}

double NodeParameters_Omega::GetBasalExpression_Beta() {
    return this->basalExpression_beta;
}

double NodeParameters_Omega::GetDecayRate_Gamma() {
    return this->decayRate_gamma;
}

double NodeParameters_Omega::GetScaleFactor_Tau() {
    return this->scaleFactor_tau;
}

std::vector<double> NodeParameters_Omega::GetWeights_W() {
    return this->v_weights_w;
}


//////////////////////////////////////////////////////////////////////////
//NeuralNetwork

NeuralNetwork::NeuralNetwork() {
	//std::cout << "Empty NodeParameter_Omega constructor called" << std::endl;
}

NeuralNetwork::NeuralNetwork(std::vector<NodeParameters_Omega*> inputNodeParameters_omega)
                            : v_nodeParameters_omega(inputNodeParameters_omega) {
    //std::cout << "Neural Network intialized with vector" << std::endl;
}

NeuralNetwork::~NeuralNetwork() {
    for (int i = 0; i < this->v_nodeParameters_omega.size(); i++) {
        delete v_nodeParameters_omega[i];
    }
}

void NeuralNetwork::AddNodeParametersToNetwork(NodeParameters_Omega * inputNodeParameters_Omega) {
    this->v_nodeParameters_omega.push_back(inputNodeParameters_Omega);
}

std::vector<NodeParameters_Omega*> NeuralNetwork::GetNodesInNetwork() {
    return this->v_nodeParameters_omega;
}

void NeuralNetwork::PrintNeuralNetwork() {
    for (int i = 0; i < this->v_nodeParameters_omega.size(); i++) {
        v_nodeParameters_omega[i]->PrintNodeParameters();
    }
}

void NeuralNetwork::SetNodesInNetwork(std::vector<NodeParameters_Omega*> v_inputNodeParameters_omega) {
    this->v_nodeParameters_omega = v_inputNodeParameters_omega;
}

int NeuralNetwork::GetNumberNodesInNetwork() {
    return this->v_nodeParameters_omega.size();
}

NodeParameters_Omega * NeuralNetwork::GetNodeFromIndex(int indexNode) {
    return this->v_nodeParameters_omega[indexNode];
}

//////////////////////////////////////////////////////////////////////////
//NeuralNetworkInitializer
NeuralNetworkInitializer::NeuralNetworkInitializer() {
    //std::cout << "Empty NeuralNetworkInitializer constructor called" << std::endl;
}

void NeuralNetworkInitializer::InitializeNeuralNetwork(NeuralNetwork * inputNeuralNetwork, TimeSeriesSet * inputTimeSeriesSet) {
    std::cout << "Initializing Neural Network" << std::endl;
    std::vector<std::string> observationNames = inputTimeSeriesSet->GetObservationNames();
    for (int i = 0; i < observationNames.size(); i++) {
        std::vector<double> weight_zero_vector(observationNames.size(), 0.0);
        NodeParameters_Omega * nodeParameters_omega = new NodeParameters_Omega(observationNames[i],
                                                                               weight_zero_vector,
                                                                               0.0, 0.0, 0.0);
        inputNeuralNetwork->AddNodeParametersToNetwork(nodeParameters_omega);
    }
}

//////////////////////////////////////////////////////////////////////////
//NodeParameter_Operator
NodeParameter_Operator::NodeParameter_Operator() {
    //std::cout << "Empty NodeParameter_Operator constructor called." << std::endl;
}

NodeParameters_Omega * NodeParameter_Operator::Addition(NodeParameters_Omega * inputNode1, NodeParameters_Omega * inputNode2, NodeParameters_Omega * outputNode) {
    //std::cout << "Adding nodes" << std::endl;
    //Name
    std::string nodeName = inputNode1->GetNodeName();

    //Weights
    std::vector<double> input1weights = inputNode1->GetWeights_W();
    std::vector<double> input2weights = inputNode2->GetWeights_W();
    int numWeights = inputNode1->GetNumberWeights();
    std::vector<double> outputWeights(numWeights, 0.0);
    assert(input1weights.size() == input2weights.size() && input2weights.size() == outputWeights.size());
    for (int i = 0; i < input1weights.size(); i++) {
        outputWeights[i] = input1weights[i] + input2weights[i];
    }

    //Beta, Gamma, and Tau
    double outputBeta = inputNode1->GetBasalExpression_Beta() + inputNode2->GetBasalExpression_Beta();
    double outputGamma = inputNode1->GetDecayRate_Gamma() + inputNode2->GetDecayRate_Gamma();
    double outputTau = inputNode1->GetScaleFactor_Tau() + inputNode2->GetScaleFactor_Tau();

    //Set for the output node.
    outputNode->SetNodeParameters(nodeName, outputWeights,
                                  outputBeta, outputGamma, outputTau);
    return outputNode;
}

NodeParameters_Omega * NodeParameter_Operator::Subtraction(NodeParameters_Omega * inputNode1, NodeParameters_Omega * inputNode2, NodeParameters_Omega * outputNode) {
    //std::cout << "Subtracting nodes" << std::endl;

    //Name
    std::string nodeName = inputNode1->GetNodeName();

    //Weights
    std::vector<double> input1weights = inputNode1->GetWeights_W();
    std::vector<double> input2weights = inputNode2->GetWeights_W();
    int numWeights = inputNode1->GetNumberWeights();
    std::vector<double> outputWeights(numWeights, 0.0);
    assert(input1weights.size() == input2weights.size() && input2weights.size() == outputWeights.size());
    for (int i = 0; i < input1weights.size(); i++) {
        outputWeights[i] = input1weights[i] - input2weights[i];
    }

    //Beta, Gamma, and Tau
    double outputBeta = inputNode1->GetBasalExpression_Beta() - inputNode2->GetBasalExpression_Beta();
    double outputGamma = inputNode1->GetDecayRate_Gamma() - inputNode2->GetDecayRate_Gamma();
    double outputTau = inputNode1->GetScaleFactor_Tau() - inputNode2->GetScaleFactor_Tau();

    //Set for the output node.
    outputNode->SetNodeParameters(nodeName, outputWeights,
                                  outputBeta, outputGamma, outputTau);
    return outputNode;
}

NodeParameters_Omega * NodeParameter_Operator::Multiplication(NodeParameters_Omega * inputNode1, double value, NodeParameters_Omega * outputNode) {
    //std::cout << "Multiplying a node by a value" << std::endl;
    //Name
    std::string nodeName = inputNode1->GetNodeName();

    //Weights
    std::vector<double> input1weights = inputNode1->GetWeights_W();
    int numWeights = inputNode1->GetNumberWeights();
    std::vector<double> outputWeights(numWeights, 0.0);
    for (int i = 0; i < input1weights.size(); i++) {
        outputWeights[i] = input1weights[i] * value;
    }

    //Beta, Gamma, and Tau
    double outputBeta = inputNode1->GetBasalExpression_Beta() * value;
    double outputGamma = inputNode1->GetDecayRate_Gamma()  * value;
    double outputTau = inputNode1->GetScaleFactor_Tau() * value;

    //Set for the output node.
    outputNode->SetNodeParameters(nodeName, outputWeights,
                                  outputBeta, outputGamma, outputTau);
    return outputNode;
}
