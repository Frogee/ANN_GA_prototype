#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <string>
#include <vector>
#include "dataUtil.h"

class NodeParameters_Omega {
	public:
        NodeParameters_Omega();
		NodeParameters_Omega(std::string inputNodeName, std::vector<double> inputWeights_w, double inputBasalExpression_beta, double inputDecayRate_gamma, double inputScaleFactor_tau);
		~NodeParameters_Omega();
		void SetNodeParameters(std::string inputNodeName, std::vector<double> inputWeights_w, double inputBasalExpression_beta, double inputDecayRate_gamma, double inputScaleFactor_tau);
        void PrintNodeParameters();
        std::string GetNodeName();
        int GetNumberWeights();
        double GetBasalExpression_Beta();
        double GetDecayRate_Gamma();
        double GetScaleFactor_Tau();
        std::vector<double> GetWeights_W();
	private:
		std::string nodeName;
		std::vector<double> v_weights_w;
		double basalExpression_beta;
		double decayRate_gamma;
		double scaleFactor_tau;
};

class NeuralNetwork {
	public:
        NeuralNetwork();
		NeuralNetwork(std::vector<NodeParameters_Omega*> inputNodeParameters_omega);
		~NeuralNetwork();
        void AddNodeParametersToNetwork(NodeParameters_Omega * inputNodeParameters_Omega);
        void SetNodesInNetwork(std::vector<NodeParameters_Omega*> v_inputNodeParameters_omega);
        std::vector<NodeParameters_Omega*> GetNodesInNetwork();
        int GetNumberNodesInNetwork();
        NodeParameters_Omega * GetNodeFromIndex(int indexNode);
        void PrintNeuralNetwork();
	private:
		std::vector<NodeParameters_Omega*> v_nodeParameters_omega;
};

class NeuralNetworkInitializer {
    public:
        NeuralNetworkInitializer();
        void InitializeNeuralNetwork(NeuralNetwork * inputNeuralNetwork, TimeSeriesSet * inputTimeSeriesSet);
};

class NodeParameter_Operator {
    public:
        NodeParameter_Operator();
        NodeParameters_Omega * Addition(NodeParameters_Omega * inputNode1, NodeParameters_Omega * inputNode2, NodeParameters_Omega * outputNode);
        NodeParameters_Omega * Subtraction(NodeParameters_Omega * inputNode1, NodeParameters_Omega * inputNode2, NodeParameters_Omega * outputNode);
        NodeParameters_Omega * Multiplication(NodeParameters_Omega * inputNode1, double value, NodeParameters_Omega * outputNode);
};
#endif
