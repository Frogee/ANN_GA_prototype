#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

#include "neuralNetwork.h"

class Population {
    public:
        Population();
        Population(std::vector<NeuralNetwork*> inputNeuralNetworkPopulation);
        NeuralNetwork* GetNetworkFromIndex(int inputNetworkIndex);
        int GetPopulationSize();
        void ReplaceIndividual(int indexOfIndividualToReplace, NeuralNetwork* inputReplacement);
        void SetFitnessValue(int indexOfFitnessValue, double fitnessValue);
        NeuralNetwork* GetMostFitIndividual();
        double GetFitnessOfMostFitIndividual();
        double GetFitnessOfLeastFitIndividual();
        void PrintPopulation();
    private:
        std::vector<NeuralNetwork*> v_neuralNetworkPopulation;
        std::vector<double> v_fitnessValues;
        std::string networkName;
};

class GeneticAlgorithm {
	public:
        GeneticAlgorithm();
        void EvolveNetwork(NeuralNetwork * inputNeuralNetwork, TimeSeriesSet * inputTimeSeriesSet);
        double CalculateNetworkFitnessVerbose(NeuralNetwork* inputNetwork, TimeSeriesSet * inputTimeSeriesSet);
    private:
        Population* InitializePopulation(NeuralNetwork* inputNeuralNetwork);
        void RestartPopulationWithMostFitIndividual(Population* inputPopulation);
        NeuralNetwork* GenerateRandomNetwork(NeuralNetwork* inputNeuralNetwork);
        NodeParameters_Omega* GenerateRandomNode(NodeParameters_Omega* inputNodeParameters_omega);
        NeuralNetwork* GenerateMutant(NeuralNetwork* inputNetwork, Population* inputPopulation);
        NodeParameters_Omega* GenerateMutant(NodeParameters_Omega* inputNode, Population* inputPopulation);
        NeuralNetwork* EvaluateCrossover(NeuralNetwork* inputNetwork, NeuralNetwork* mutantNetwork);
        NodeParameters_Omega* EvaluateCrossover(NodeParameters_Omega* inputNode, NodeParameters_Omega* mutantNode);
        std::pair<bool,double> EvaluateFitness(NeuralNetwork* inputNetwork, NeuralNetwork* crossoverNetwork, TimeSeriesSet * inputTimeSeriesSet);
        std::pair<bool,double> EvaluateFitness(NodeParameters_Omega* inputNode, NodeParameters_Omega* crossoverNode, TimeSeriesSet * inputTimeSeriesSet);
        double SigmoidalTransfer(double inputTransfer);
        double CalculateNetworkFitness(NeuralNetwork* inputNetwork, TimeSeriesSet * inputTimeSeriesSet);
        double CalculateNodeFitness(NodeParameters_Omega* inputNode, TimeSeriesSet * inputTimeSeriesSet);
        double CalculatePenaltyTerm(std::vector<double> inputNodeWeights);
        void ConstrainNodeParametersToRanges(NodeParameters_Omega* inputNode);
        int G_MAX; //These are initialized with the constructor.
        int POP_SIZE;
        double SCALING_FACTOR_F;
        double CROSSOVER_FACTOR_CF;
        double FITNESS_THRESHOLD_DELTA;
        double PRUNING_CONSTANT_C;
        double MAX_ALLOWED_INTERACTIONS_I;
        std::pair<double, double> WEIGHT_RANGE;
        std::pair<double, double> BETA_RANGE;
        std::pair<double, double> GAMMA_RANGE;
        std::pair<double, double> TAU_RANGE;
};


#endif
