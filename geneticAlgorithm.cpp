#include <cassert>
#include <math.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include "geneticAlgorithm.h"

#define NETWORK_SIZE 4   ///Currently the network size has to be predefined for the preprocessor
                         ///I don't know how to get it to work at runtime due to odeint's requirements for the system of equations
                         ///TODO: See if this can be solved so that the system of equations can be defined at runtime based on input data.

//////////////////////////////////////////////////////////////////////////
//Genetic Algorithm

GeneticAlgorithm::GeneticAlgorithm() : G_MAX(1000000), POP_SIZE(1000),
                                        SCALING_FACTOR_F(0.5), CROSSOVER_FACTOR_CF(0.9),
                                        FITNESS_THRESHOLD_DELTA(0.001), PRUNING_CONSTANT_C(10),
                                        MAX_ALLOWED_INTERACTIONS_I(3),
                                        WEIGHT_RANGE(-30.0, 30.0), BETA_RANGE(-10.0, 10.0),
                                        GAMMA_RANGE(-10.0, 10.0), TAU_RANGE(0.0, 20.0) {


    std::cout << "Genetic Algorithm Parameters are: " << std::endl <<
    "\tG_MAX:\t" << this->G_MAX << std::endl <<
    "\tPOP_SIZE:\t" << this->POP_SIZE << std::endl <<
    "\tSCALING_FACTOR_F:\t" << this->SCALING_FACTOR_F << std::endl <<
    "\tCROSSOVER_FACTOR_CF:\t" << this->CROSSOVER_FACTOR_CF << std::endl <<
    "\tFITNESS_THRESHOLD_DELTA:\t" << this->FITNESS_THRESHOLD_DELTA << std::endl <<
    "\tPRUNING_CONSTANT_C:\t" << this->PRUNING_CONSTANT_C << std::endl <<
    "\tMAX_ALLOWED_INTERACTIONS_I:\t" << this->MAX_ALLOWED_INTERACTIONS_I << std::endl <<
    "\tWEIGHT_RANGE:\t" << "[" << this->WEIGHT_RANGE.first << "," << this->WEIGHT_RANGE.second << "]" << std::endl <<
    "\tBETA_RANGE:\t" << "[" << this->BETA_RANGE.first << "," << this->BETA_RANGE.second << "]" <<std::endl <<
    "\tGAMMA_RANGE:\t" << "[" << this->GAMMA_RANGE.first << "," << this->GAMMA_RANGE.second << "]" <<std::endl <<
    "\tTAU_RANGE:\t" << "[" << this->TAU_RANGE.first << "," << this->TAU_RANGE.second << "]" <<std::endl;
}

NeuralNetwork* GeneticAlgorithm::GenerateRandomNetwork(NeuralNetwork* inputNeuralNetwork) {
    std::vector<NodeParameters_Omega*> nodesInInputNetwork = inputNeuralNetwork->GetNodesInNetwork();
    std::vector<NodeParameters_Omega*> nodesInRandomNetwork;
    for (int i = 0; i < nodesInInputNetwork.size(); i++) {
        nodesInRandomNetwork.push_back(GenerateRandomNode(nodesInInputNetwork[i]));
    }
    NeuralNetwork * randomNetwork = new NeuralNetwork(nodesInRandomNetwork);
    return randomNetwork;
}

NodeParameters_Omega* GeneticAlgorithm::GenerateRandomNode(NodeParameters_Omega* inputNodeParameters_omega) {
    //std::cout << "Generating random node" << std::endl;

    //Setup the random number generator
    //Not sure how to decide on the ranges; approximating Noman et al. for now.
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 randomGenerator(seed);
    std::uniform_real_distribution<double> distributionWeights(this->WEIGHT_RANGE.first, this->WEIGHT_RANGE.second);
    std::uniform_real_distribution<double> distributionBasalExpression(BETA_RANGE.first, BETA_RANGE.second);
    std::uniform_real_distribution<double> distributionDecayRate(GAMMA_RANGE.first, GAMMA_RANGE.second);
    std::uniform_real_distribution<double> distributionScaleFactor(TAU_RANGE.first, TAU_RANGE.second);

    std::string nodeName = "NULL";

    std::vector<double> v_weights(inputNodeParameters_omega->GetNumberWeights(), 0.0);

    double basalExpression = 0.0;
    double decayRate = 0.0;
    double scaleFactor = 0.0;

    for (int j = 0; j < v_weights.size(); j++) {
        v_weights[j] = distributionWeights(randomGenerator);
    }

    nodeName = inputNodeParameters_omega->GetNodeName();
    basalExpression = distributionBasalExpression(randomGenerator);
    decayRate = distributionDecayRate(randomGenerator);
    scaleFactor = distributionScaleFactor(randomGenerator);

    NodeParameters_Omega * newNode = new NodeParameters_Omega(nodeName,
                                                                v_weights,
                                                                basalExpression,
                                                                decayRate,
                                                                scaleFactor);
    return newNode;
}

NeuralNetwork* GeneticAlgorithm::GenerateMutant(NeuralNetwork* inputNetwork, Population* inputPopulation) {
        //std::cout << "Generating mutant" << std::endl;

    //The mutation operation is y_iG = x_jG + F(x_kG - xlG) where i != j != k != l
    //We generate 3 random indices such that non equal the other and non equal the original input.
    //First find the original index.
    int indexOfInputNetwork = 0;
    int indexOfIndividualJ = 0;
    int indexOfIndividualK = 0;
    int indexOfIndividualL = 0;
    for (int i = 0; i < inputPopulation->GetPopulationSize(); i++) {
        NeuralNetwork * currentNetwork = inputPopulation->GetNetworkFromIndex(i);
        if (inputNetwork == currentNetwork) {
            //std::cout << "Node pointers match on index " << i << std::endl;
            indexOfInputNetwork = i;
            break;
        }
    }
    //Second, get 3 additional unique indices
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 randomGenerator(seed);
    std::uniform_int_distribution<int> distributionIndices(0, inputPopulation->GetPopulationSize() - 1);
    while (indexOfInputNetwork == indexOfIndividualJ ||
           indexOfInputNetwork == indexOfIndividualK ||
           indexOfInputNetwork == indexOfIndividualL ||
           indexOfIndividualJ == indexOfIndividualK ||
           indexOfIndividualJ == indexOfIndividualL ||
           indexOfIndividualK == indexOfIndividualL) {
        indexOfIndividualJ = distributionIndices(randomGenerator);
        indexOfIndividualK = distributionIndices(randomGenerator);
        indexOfIndividualL = distributionIndices(randomGenerator);
    }
    //For each node, apply the mutation operation
    std::vector<NodeParameters_Omega*> mutantNodes;
    NeuralNetwork * individualJ = inputPopulation->GetNetworkFromIndex(indexOfIndividualJ);
    NeuralNetwork * individualK = inputPopulation->GetNetworkFromIndex(indexOfIndividualK);
    NeuralNetwork * individualL = inputPopulation->GetNetworkFromIndex(indexOfIndividualL);
    for (int indexNode = 0; indexNode < inputNetwork->GetNumberNodesInNetwork(); indexNode++) {

        NodeParameters_Omega * subtractionResult = new NodeParameters_Omega;
        NodeParameters_Omega * multiplicationResult = new NodeParameters_Omega;
        NodeParameters_Omega * additionResult = new NodeParameters_Omega;

        NodeParameter_Operator * nodeOperator = new NodeParameter_Operator;

        NodeParameters_Omega * individualJnode = individualJ->GetNodeFromIndex(indexNode);
        NodeParameters_Omega * individualKnode = individualK->GetNodeFromIndex(indexNode);
        NodeParameters_Omega * individualLnode = individualL->GetNodeFromIndex(indexNode);

        nodeOperator->Subtraction(individualKnode, individualLnode, subtractionResult);
        nodeOperator->Multiplication(subtractionResult, this->SCALING_FACTOR_F, multiplicationResult);
        nodeOperator->Addition(multiplicationResult, individualJnode, additionResult);

        this->ConstrainNodeParametersToRanges(additionResult);
        NodeParameters_Omega * mutantNode = additionResult;
        mutantNodes.push_back(mutantNode);

        /*
        std::cout << std::endl << std::endl << "individualJ\t";
        individualJ->PrintNodeParameters();
        std::cout << std::endl << std::endl << "individualK\t";
        individualK->PrintNodeParameters();
        std::cout << std::endl << std::endl << "individualL\t";
        individualL->PrintNodeParameters();
        std::cout << std::endl << std::endl << "Subtraction Result:\t";
        subtractionResult->PrintNodeParameters();
        std::cout << std::endl << std::endl << "Multiplication Result:\t";
        multiplicationResult->PrintNodeParameters();
        std::cout << std::endl << std::endl << "Addition Result:\t";
        additionResult->PrintNodeParameters();
        std::cout << std::endl << std::endl << "The mutant is:\t";
        mutant->PrintNodeParameters();
        */

        delete subtractionResult;
        delete multiplicationResult;
        delete nodeOperator;
    }
    NeuralNetwork * mutantNetwork = new NeuralNetwork(mutantNodes);
    return mutantNetwork;
}

NodeParameters_Omega* GeneticAlgorithm::GenerateMutant(NodeParameters_Omega* inputNode, Population* inputPopulation) {
    //std::cout << "Generating mutant" << std::endl;

    //The mutation operation is y_iG = x_jG + F(x_kG - xlG) where i != j != k != l
    //We generate 3 random indices such that non equal the other and non equal the original input.
    //First find the original index.
    /*
    int indexOfInputNode = 0;
    int indexOfIndividualJ = 0;
    int indexOfIndividualK = 0;
    int indexOfIndividualL = 0;
    for (int i = 0; i < inputPopulation->GetPopulationSize(); i++) {
        NodeParameters_Omega * currentNode = inputPopulation->GetNodeFromIndex(i);
        if (inputNode == currentNode) {
            //std::cout << "Node pointers match on index " << i << std::endl;
            indexOfInputNode = i;
            break;
        }
    }
    //Second, get 3 additional unique indices
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 randomGenerator(seed);
    std::uniform_int_distribution<int> distributionIndices(0, inputPopulation->GetPopulationSize() - 1);
    while (indexOfInputNode == indexOfIndividualJ ||
           indexOfInputNode == indexOfIndividualK ||
           indexOfInputNode == indexOfIndividualL ||
           indexOfIndividualJ == indexOfIndividualK ||
           indexOfIndividualJ == indexOfIndividualL ||
           indexOfIndividualK == indexOfIndividualL) {
        indexOfIndividualJ = distributionIndices(randomGenerator);
        indexOfIndividualK = distributionIndices(randomGenerator);
        indexOfIndividualL = distributionIndices(randomGenerator);
    }
    //Apply the mutation operation
    NodeParameters_Omega * subtractionResult = new NodeParameters_Omega;
    NodeParameters_Omega * multiplicationResult = new NodeParameters_Omega;
    NodeParameters_Omega * additionResult = new NodeParameters_Omega;
    NodeParameters_Omega * mutant;
    NodeParameter_Operator * nodeOperator = new NodeParameter_Operator;

    NodeParameters_Omega * individualJ = inputPopulation->GetNodeFromIndex(indexOfIndividualJ);
    NodeParameters_Omega * individualK = inputPopulation->GetNodeFromIndex(indexOfIndividualK);
    NodeParameters_Omega * individualL = inputPopulation->GetNodeFromIndex(indexOfIndividualL);

    nodeOperator->Subtraction(individualK, individualL, subtractionResult);
    nodeOperator->Multiplication(subtractionResult, this->SCALING_FACTOR_F, multiplicationResult);
    nodeOperator->Addition(multiplicationResult, individualJ, additionResult);

    this->ConstrainNodeParametersToRanges(additionResult);
    mutant = additionResult;


    std::cout << std::endl << std::endl << "individualJ\t";
    individualJ->PrintNodeParameters();
    std::cout << std::endl << std::endl << "individualK\t";
    individualK->PrintNodeParameters();
    std::cout << std::endl << std::endl << "individualL\t";
    individualL->PrintNodeParameters();
    std::cout << std::endl << std::endl << "Subtraction Result:\t";
    subtractionResult->PrintNodeParameters();
    std::cout << std::endl << std::endl << "Multiplication Result:\t";
    multiplicationResult->PrintNodeParameters();
    std::cout << std::endl << std::endl << "Addition Result:\t";
    additionResult->PrintNodeParameters();
    std::cout << std::endl << std::endl << "The mutant is:\t";
    mutant->PrintNodeParameters();


    delete subtractionResult;
    delete multiplicationResult;
    delete nodeOperator;

    return mutant;
        */
}

Population* GeneticAlgorithm::InitializePopulation(NeuralNetwork* inputNeuralNetwork) {
    std::cout << "Initializing population" << std::endl;
    std::vector<NeuralNetwork*> v_neuralNetworkPopulation;

    for (int i = 0; i <= this->POP_SIZE - 1; i++) {
        /*
        if (i == 0) {
            std::vector<NodeParameters_Omega*> v_modelParameters;
            assert(inputNeuralNetwork->GetNumberNodesInNetwork() == 4);
            //Seed with perfect node parameters for for genes 1, 2, 3, and 4 for debugging.
            std::vector<double> perfectWeights {20.0, 5.0, 0.0, 0.0};
            std::string nodeName = (inputNeuralNetwork->GetNodeFromIndex(0))->GetNodeName();
            NodeParameters_Omega* perfectNode = new NodeParameters_Omega(nodeName,
                                                                             perfectWeights,
                                                                             0.0, 1.0, 10.0);
            v_modelParameters.push_back(perfectNode);
            //Seed with perfect node for gene 2 for debugging.
            perfectWeights = {25.0, -5.0, -17.0, 0.0};
            nodeName = (inputNeuralNetwork->GetNodeFromIndex(1))->GetNodeName();
            perfectNode = new NodeParameters_Omega(nodeName,
                                                                             perfectWeights,
                                                                             -5.0, 1.0, 5.0);
            v_modelParameters.push_back(perfectNode);
            //Seed with perfect node for gene 3 for debugging.
            perfectWeights = {0.0, 10.0, 20.0, -20.0};
            nodeName = (inputNeuralNetwork->GetNodeFromIndex(2))->GetNodeName();
            perfectNode = new NodeParameters_Omega(nodeName,
                                                                             perfectWeights,
                                                                             -5.0, 1.0, 5.0);
            v_modelParameters.push_back(perfectNode);
            //Seed with perfect node for gene 4 for debugging.
            perfectWeights = {0.0, 0.0, 10.0, -5.0};
            nodeName = (inputNeuralNetwork->GetNodeFromIndex(3))->GetNodeName();
            perfectNode = new NodeParameters_Omega(nodeName,
                                                                             perfectWeights,
                                                                             0.0, 1.0, 10.0);
            v_modelParameters.push_back(perfectNode);
            NeuralNetwork * perfectNetwork = new NeuralNetwork(v_modelParameters);
            v_neuralNetworkPopulation.push_back(perfectNetwork);
            continue;
            } */
        v_neuralNetworkPopulation.push_back(this->GenerateRandomNetwork(inputNeuralNetwork));
    }
    Population * population = new Population(v_neuralNetworkPopulation);
    population->PrintPopulation();
    return population;
}

void GeneticAlgorithm::RestartPopulationWithMostFitIndividual(Population* inputPopulation) {
    std::cout << "Restarting the population" << std::endl;
    int populationSize = inputPopulation->GetPopulationSize();
    NeuralNetwork* mostFitNetwork = inputPopulation->GetMostFitIndividual();
    //mostFitNode->PrintNodeParameters();
    for (int i = 0; i < populationSize; i++) {
        inputPopulation->ReplaceIndividual(i, this->GenerateRandomNetwork(mostFitNetwork));
    }
    inputPopulation->ReplaceIndividual(0, mostFitNetwork);
}

void GeneticAlgorithm::EvolveNetwork(NeuralNetwork * inputNeuralNetwork, TimeSeriesSet * inputTimeSeriesSet) {
    std::cout << "Beginning network evolution" << std::endl;
    Population * population;

    population = this->InitializePopulation(inputNeuralNetwork);
    //population->PrintPopulation();

    double currentGenMaxFitness = 0.0;
    double currentGenMinFitness = 0.0;
    double fitnessDifference = 0.0;
    for (int generation = 0; generation < G_MAX; generation++) {
        for (int i = 0; i < population->GetPopulationSize(); i++) {
            NeuralNetwork * currentNetwork = population->GetNetworkFromIndex(i);
            NeuralNetwork * mutantNetwork = this->GenerateMutant(currentNetwork, population);
            NeuralNetwork * crossoverNetwork = this->EvaluateCrossover(currentNetwork, mutantNetwork);
            delete mutantNetwork;

            std::pair<bool,double> parentIsMoreFit = this->EvaluateFitness(currentNetwork, crossoverNetwork, inputTimeSeriesSet);

            if (parentIsMoreFit.first) {
                //Keep the parent node the same
                //Set the fitness value
                double fitness = parentIsMoreFit.second;
                population->SetFitnessValue(i, fitness);
                delete crossoverNetwork;
            }
            else {
                population->ReplaceIndividual(i, crossoverNetwork);
                double fitness = parentIsMoreFit.second;
                population->SetFitnessValue(i, fitness);
            }
        }
        currentGenMaxFitness = population->GetFitnessOfMostFitIndividual();   //This is actually the smallest number
        currentGenMinFitness = population->GetFitnessOfLeastFitIndividual();  //This is actually the largest number
        fitnessDifference = currentGenMinFitness - currentGenMaxFitness;
        if (fitnessDifference < (this->FITNESS_THRESHOLD_DELTA * currentGenMaxFitness)) {
            std::cout << "Restarting the population." << std::endl;
            //population->PrintPopulation();
            this->RestartPopulationWithMostFitIndividual(population);
            //std::cout << "New population: " << std::endl;
            //population->PrintPopulation();
        }
        //std::cout << "The population after generation: " << generation << std::endl;
        //population->PrintPopulation();
        std::cout << "\tThe most fit network after generation: " << generation << std::endl;
        (population->GetMostFitIndividual())->PrintNeuralNetwork();
        std::cout << "\tFitness: " << population->GetFitnessOfMostFitIndividual() << std::endl;
    }
    //population->PrintPopulation();
    std::vector<NodeParameters_Omega*> v_nodesInNetwork = (population->GetMostFitIndividual())->GetNodesInNetwork();;
    inputNeuralNetwork->SetNodesInNetwork(v_nodesInNetwork);
}

NeuralNetwork * GeneticAlgorithm::EvaluateCrossover(NeuralNetwork* inputNetwork, NeuralNetwork* mutantNetwork) {
       //std::cout << "Evaluating crossover" << std::endl;

    std::vector<NodeParameters_Omega*> crossoverNodes;

    for (int indexNode = 0; indexNode < inputNetwork->GetNumberNodesInNetwork(); indexNode++) {
        NodeParameters_Omega * inputNode = inputNetwork->GetNodeFromIndex(indexNode);
        NodeParameters_Omega * mutantNode = mutantNetwork->GetNodeFromIndex(indexNode);

        double randomCrossoverFactorValue = 0.0;
        NodeParameters_Omega * crossoverResult = new NodeParameters_Omega;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        static std::mt19937 randomGenerator(seed);
        std::uniform_real_distribution<double> distribution(0, 1.0);

        //Name
        std::string nodeName = inputNode->GetNodeName();

        //Weights
        std::vector<double> inputWeights = inputNode->GetWeights_W();
        std::vector<double> mutantWeights = mutantNode->GetWeights_W();
        int numWeights = inputNode->GetNumberWeights();
        std::vector<double> outputWeights(numWeights, 0.0);
        assert(inputWeights.size() == mutantWeights.size() && inputWeights.size() == outputWeights.size());
        for (int i = 0; i < inputWeights.size(); i++) {
            randomCrossoverFactorValue = distribution(randomGenerator);
            if (randomCrossoverFactorValue <= this->CROSSOVER_FACTOR_CF) {
                outputWeights[i] = inputWeights[i];
            }
            else {
                outputWeights[i] = mutantWeights[i];
            }
        }

        //Beta, Gamma, and Tau
        randomCrossoverFactorValue = distribution(randomGenerator);
        double outputBeta;
        if (randomCrossoverFactorValue <= this->CROSSOVER_FACTOR_CF) {
            outputBeta = inputNode->GetBasalExpression_Beta();
        }
        else {
            outputBeta = mutantNode->GetBasalExpression_Beta();
        }

        randomCrossoverFactorValue = distribution(randomGenerator);
        double outputGamma;
        if (randomCrossoverFactorValue <= this->CROSSOVER_FACTOR_CF) {
            outputGamma = inputNode->GetDecayRate_Gamma();
        }
        else {
            outputGamma = mutantNode->GetDecayRate_Gamma();
        }

        randomCrossoverFactorValue = distribution(randomGenerator);
        double outputTau;
        if (randomCrossoverFactorValue <= this->CROSSOVER_FACTOR_CF) {
            outputTau = inputNode->GetScaleFactor_Tau();
        }
        else {
            outputTau = mutantNode->GetScaleFactor_Tau();
        }

        //Set for the output node.
        crossoverResult->SetNodeParameters(nodeName, outputWeights,
                                      outputBeta, outputGamma, outputTau);

        crossoverNodes.push_back(crossoverResult);
    }
    NeuralNetwork * crossoverResult = new NeuralNetwork(crossoverNodes);
    return crossoverResult;
}

NodeParameters_Omega* GeneticAlgorithm::EvaluateCrossover(NodeParameters_Omega* inputNode, NodeParameters_Omega* mutantNode) {
    //std::cout << "Evaluating crossover" << std::endl;
    double randomCrossoverFactorValue = 0.0;
    NodeParameters_Omega * crossoverResult = new NodeParameters_Omega;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 randomGenerator(seed);
    std::uniform_real_distribution<double> distribution(0, 1.0);

    //Name
    std::string nodeName = inputNode->GetNodeName();

    //Weights
    std::vector<double> inputWeights = inputNode->GetWeights_W();
    std::vector<double> mutantWeights = mutantNode->GetWeights_W();
    int numWeights = inputNode->GetNumberWeights();
    std::vector<double> outputWeights(numWeights, 0.0);
    assert(inputWeights.size() == mutantWeights.size() && inputWeights.size() == outputWeights.size());
    for (int i = 0; i < inputWeights.size(); i++) {
        randomCrossoverFactorValue = distribution(randomGenerator);
        if (randomCrossoverFactorValue <= this->CROSSOVER_FACTOR_CF) {
            outputWeights[i] = inputWeights[i];
        }
        else {
            outputWeights[i] = mutantWeights[i];
        }
    }

    //Beta, Gamma, and Tau
    randomCrossoverFactorValue = distribution(randomGenerator);
    double outputBeta;
    if (randomCrossoverFactorValue <= this->CROSSOVER_FACTOR_CF) {
        outputBeta = inputNode->GetBasalExpression_Beta();
    }
    else {
        outputBeta = mutantNode->GetBasalExpression_Beta();
    }

    randomCrossoverFactorValue = distribution(randomGenerator);
    double outputGamma;
    if (randomCrossoverFactorValue <= this->CROSSOVER_FACTOR_CF) {
        outputGamma = inputNode->GetDecayRate_Gamma();
    }
    else {
        outputGamma = mutantNode->GetDecayRate_Gamma();
    }

    randomCrossoverFactorValue = distribution(randomGenerator);
    double outputTau;
    if (randomCrossoverFactorValue <= this->CROSSOVER_FACTOR_CF) {
        outputTau = inputNode->GetScaleFactor_Tau();
    }
    else {
        outputTau = mutantNode->GetScaleFactor_Tau();
    }

    //Set for the output node.
    crossoverResult->SetNodeParameters(nodeName, outputWeights,
                                  outputBeta, outputGamma, outputTau);
    return crossoverResult;
}

std::pair<bool,double> GeneticAlgorithm::EvaluateFitness(NeuralNetwork* inputNetwork, NeuralNetwork* crossoverNetwork, TimeSeriesSet * inputTimeSeriesSet) {
    std::pair<bool,double> parentIsMoreFit;

    double inputNetworkFitness;
    double crossoverNetworkFitness;
    inputNetworkFitness = this->CalculateNetworkFitness(inputNetwork, inputTimeSeriesSet);
    crossoverNetworkFitness = this->CalculateNetworkFitness(crossoverNetwork, inputTimeSeriesSet);
    //std::cout << "Input Network fitness: " << inputNetworkFitness << "\tCrossover Network fitness: " << crossoverNetworkFitness << std::endl;

    if (inputNetworkFitness < crossoverNetworkFitness) {
        parentIsMoreFit.first = true;
        parentIsMoreFit.second = inputNetworkFitness;
    }
    else {
        parentIsMoreFit.first = false;
        parentIsMoreFit.second = crossoverNetworkFitness;
    }
    return parentIsMoreFit;
}

std::pair<bool,double> GeneticAlgorithm::EvaluateFitness(NodeParameters_Omega* inputNode, NodeParameters_Omega* crossoverNode, TimeSeriesSet * inputTimeSeriesSet) {
    //std::cout << "Evaluating fitness" << std::endl; //Returns true if inputNode is more fit
    std::pair<bool,double> parentIsMoreFit;

    double inputNodeFitness;
    double crossoverNodeFitness;
    inputNodeFitness = this->CalculateNodeFitness(inputNode, inputTimeSeriesSet);
    crossoverNodeFitness = this->CalculateNodeFitness(crossoverNode, inputTimeSeriesSet);
    //std::cout << "Input node fitness: " << inputNodeFitness << "\tCrossover Node fitness: " << crossoverNodeFitness << std::endl;

    if (inputNodeFitness < crossoverNodeFitness) {
        parentIsMoreFit.first = true;
        parentIsMoreFit.second = inputNodeFitness;
    }
    else {
        parentIsMoreFit.first = false;
        parentIsMoreFit.second = crossoverNodeFitness;
    }
    return parentIsMoreFit;
}

double GeneticAlgorithm::SigmoidalTransfer(double inputTransfer) {
    return 1.0 / (1.0 + exp(-inputTransfer));
}

///TODO: Cleanup this mess.
///All of these declarations are for GeneticAlgorithm::CalculateNetworkFitness
std::vector<double> v_tau;
std::vector<double> v_beta;
std::vector<double> v_gamma; //aka lamda.
std::vector< std::vector <double> > vv_weights;
std::vector< std::vector <double> > vv_systemResults;
std::vector<double> v_timePointResults;
double sumWeights = 0.0;
double SigmoidFunction(double inputTransfer) {
    return 1.0 / (1.0 + exp(-inputTransfer));
}

void RNNsystem( const boost::array< double, NETWORK_SIZE > &x, boost::array< double, NETWORK_SIZE > &dxdt, double t) {
    for (int i = 0; i < NETWORK_SIZE; i++) {
        sumWeights = 0.0;
        for (int j = 0; j < NETWORK_SIZE; j++) {
            sumWeights = sumWeights + (vv_weights[i][j] * x[j]);
        }
        dxdt[i] = (1.0 / v_tau[i]) * (SigmoidFunction(sumWeights + v_beta[i]) - v_gamma[i] * x[i]);
    }
}

void push_back_timepoint( const boost::array< double, NETWORK_SIZE > &x, const double t) {
    v_timePointResults.clear();
    for (int i = 0; i < NETWORK_SIZE; i++) {
        v_timePointResults.push_back(x[i]);
    }
    vv_systemResults.push_back(v_timePointResults);
}

double GeneticAlgorithm::CalculateNetworkFitness(NeuralNetwork* inputNetwork, TimeSeriesSet * inputTimeSeriesSet) {
    assert(inputNetwork->GetNumberNodesInNetwork() == NETWORK_SIZE);
    int numberNodesInNetwork = inputNetwork->GetNumberNodesInNetwork();
    //Set the network parameters for solution of the network
    ///All of these declarations come fro above GeneticAlgorithm::CalculateNetworkFitness
    vv_weights.clear(); v_beta.clear(); v_tau.clear(); v_gamma.clear();
    for (int nodeIndex = 0; nodeIndex < numberNodesInNetwork; nodeIndex++) {
        NodeParameters_Omega* currentNode = inputNetwork->GetNodeFromIndex(nodeIndex);
        std::vector<double> v_weights = currentNode->GetWeights_W();
        vv_weights.push_back(v_weights);
        v_beta.push_back(currentNode->GetBasalExpression_Beta());
        v_tau.push_back(currentNode->GetScaleFactor_Tau());
        v_gamma.push_back(currentNode->GetDecayRate_Gamma());
    }
    //Solve and Fit for each time series set, calculate the difference, and sum those differences.
    double intermediateFitness = 0.0;
    int numberOfTimeSeriesInSet = inputTimeSeriesSet->GetNumberOfTimeSeriesInSet();
    int numberOfTimePointsInSeries = 0;
    for (int indexTimeSeries = 0; indexTimeSeries < numberOfTimeSeriesInSet; indexTimeSeries++) {
        TimeSeries * currentTimeSeries = inputTimeSeriesSet->GetTimeSeriesFromIndex(indexTimeSeries);
        if (indexTimeSeries == 0) { numberOfTimePointsInSeries = currentTimeSeries->GetNumberOfTimePointsInSeries(); }
        //Get the initial conditions
        TimePoint * initialTimePoint = currentTimeSeries->GetTimePointFromIndex(0);
        boost::array< double, NETWORK_SIZE > initialConditions;
        for (int observationIndex = 0; observationIndex < numberNodesInNetwork; observationIndex++) {
            initialConditions[observationIndex] = initialTimePoint->GetObservationFromIndex(observationIndex);
        }
        vv_systemResults.clear();
        boost::numeric::odeint::runge_kutta4< boost::array< double, NETWORK_SIZE > > rk4;  ///The stepper for the numerical solution.
        double initialTimeToEvaluate = 0.0;
        double finalTimeToEvaluate = numberOfTimePointsInSeries - 1;
        double timeStep = 1.0;
        boost::numeric::odeint::integrate_const( rk4, RNNsystem, initialConditions, initialTimeToEvaluate, finalTimeToEvaluate, timeStep, push_back_timepoint);
        //std::cout << "The results of the system for the following network:" << std::endl;
        //inputNetwork->PrintNeuralNetwork();
        for (int i = 0; i < vv_systemResults.size(); i++) {
            //std::cout << "TimePoint: " << i << "\t";
            for (int j = 0; j < vv_systemResults[i].size(); j++) {
                double calculated = vv_systemResults[i][j];
                double observed = inputTimeSeriesSet->GetObservationFromCoordinates(indexTimeSeries, i, j);
                double difference = calculated - observed;
                double differenceSquared = difference * difference;
                //std::cout << "Calculated: " << vv_systemResults[i][j] << " Observed: " <<
                //inputTimeSeriesSet->GetObservationFromCoordinates(indexTimeSeries, i, j) <<
                //" Difference: " << vv_systemResults[i][j] - inputTimeSeriesSet->GetObservationFromCoordinates(indexTimeSeries, i, j) << std::endl;
                intermediateFitness = intermediateFitness + differenceSquared;
            }
            //std::cout << std::endl;
        }
    }
    double meanSquaredError = (1.0 / (numberOfTimeSeriesInSet * numberOfTimePointsInSeries)) * intermediateFitness;
    if (isnan(meanSquaredError)) { meanSquaredError = INFINITY;} //set it arbitrarily high if there was no numerical solution.
    //std::cout << "\tMeanSquaredError: " << meanSquaredError << std::endl;
    return meanSquaredError;
}

double GeneticAlgorithm::CalculateNetworkFitnessVerbose(NeuralNetwork* inputNetwork, TimeSeriesSet * inputTimeSeriesSet) {
    assert(inputNetwork->GetNumberNodesInNetwork() == NETWORK_SIZE);
    int numberNodesInNetwork = inputNetwork->GetNumberNodesInNetwork();
    //Set the network parameters for solution of the network
    ///All of these declarations come fro above GeneticAlgorithm::CalculateNetworkFitness
    vv_weights.clear(); v_beta.clear(); v_tau.clear(); v_gamma.clear();
    for (int nodeIndex = 0; nodeIndex < numberNodesInNetwork; nodeIndex++) {
        NodeParameters_Omega* currentNode = inputNetwork->GetNodeFromIndex(nodeIndex);
        std::vector<double> v_weights = currentNode->GetWeights_W();
        vv_weights.push_back(v_weights);
        v_beta.push_back(currentNode->GetBasalExpression_Beta());
        v_tau.push_back(currentNode->GetScaleFactor_Tau());
        v_gamma.push_back(currentNode->GetDecayRate_Gamma());
    }
    //Solve and Fit for each time series set, calculate the difference, and sum those differences.
    double intermediateFitness = 0.0;
    int numberOfTimeSeriesInSet = inputTimeSeriesSet->GetNumberOfTimeSeriesInSet();
    int numberOfTimePointsInSeries = 0;
    for (int indexTimeSeries = 0; indexTimeSeries < numberOfTimeSeriesInSet; indexTimeSeries++) {
        std::cout << "TimeSeriesSet" << indexTimeSeries << "\tCalculated\tObserved\tDifferenceSquared" << std::endl;
        TimeSeries * currentTimeSeries = inputTimeSeriesSet->GetTimeSeriesFromIndex(indexTimeSeries);
        if (indexTimeSeries == 0) { numberOfTimePointsInSeries = currentTimeSeries->GetNumberOfTimePointsInSeries(); }
        //Get the initial conditions
        TimePoint * initialTimePoint = currentTimeSeries->GetTimePointFromIndex(0);
        boost::array< double, NETWORK_SIZE > initialConditions;
        for (int observationIndex = 0; observationIndex < numberNodesInNetwork; observationIndex++) {
            initialConditions[observationIndex] = initialTimePoint->GetObservationFromIndex(observationIndex);
        }
        vv_systemResults.clear();
        boost::numeric::odeint::runge_kutta4< boost::array< double, NETWORK_SIZE > > rk4;  ///The stepper for the numerical solution.
        double initialTimeToEvaluate = 0.0;
        double finalTimeToEvaluate = numberOfTimePointsInSeries - 1;
        double timeStep = 1.0;
        boost::numeric::odeint::integrate_const( rk4, RNNsystem, initialConditions, initialTimeToEvaluate, finalTimeToEvaluate, timeStep, push_back_timepoint);
        //std::cout << "The results of the system for the following network:" << std::endl;
        //inputNetwork->PrintNeuralNetwork();
        for (int i = 0; i < vv_systemResults.size(); i++) {
            std::cout << "TimePoint" << i << "\t";
            for (int j = 0; j < vv_systemResults[i].size(); j++) {
                double calculated = vv_systemResults[i][j];
                double observed = inputTimeSeriesSet->GetObservationFromCoordinates(indexTimeSeries, i, j);
                double difference = calculated - observed;
                double differenceSquared = difference * difference;
                std::cout << calculated << "\t" <<
                observed << "\t" << differenceSquared << "\t";
                intermediateFitness = intermediateFitness + differenceSquared;
            }
            std::cout << std::endl;
        }
    }
    double meanSquaredError = (1.0 / (numberOfTimeSeriesInSet * numberOfTimePointsInSeries)) * intermediateFitness;
    if (isnan(meanSquaredError)) { meanSquaredError = INFINITY;} //set it arbitrarily high if there was no numerical solution.
    std::cout << "\tMeanSquaredError: " << meanSquaredError << std::endl;
    return meanSquaredError;
}

double GeneticAlgorithm::CalculateNodeFitness(NodeParameters_Omega* inputNode, TimeSeriesSet * inputTimeSeriesSet) {
    //Here we approximate the value of e(t) using e(t-1) + De(t-1).
    int numberOfTimeSeriesInSet = inputTimeSeriesSet->GetNumberOfTimeSeriesInSet();
    assert(numberOfTimeSeriesInSet > 0);
    TimeSeries * currentTimeSeries = NULL;
    int numberOfTimePointsInSeries = 0;
    TimePoint * currentTimePoint = NULL;
    TimePoint * previousTimePoint = NULL;
    std::vector<double> v_nodeWeights;
    double calculatedExpressionOfNode = 0.0;
    double intermediateCalculationRateOfChangeOfNode = 0.0;
    double summationOfExpressionTimesWeight = 0.0;
    double observedExpressionOfNode = 0.0;
    double firstTimePoint = 0.0;
    double observationValue = 0.0;
    double weightValue = 0.0;
    double differenceBetweenCalcAndObserved = 0.0;
    double squareDifferenceBetweenCalcAndObserved = 0.0;
    double meanSquareErrorIntermediate = 0.0;
    double meanSquareError = 0.0;
    double penaltyTerm = 0.0;

    //Determine which observations the current node corresponds to.
    int OBSERVATION_INDEX = -1;
    std::vector<std::string> observationNames = inputTimeSeriesSet->GetObservationNames();
    for (int i = 0; i < observationNames.size(); i++) {
        if (inputNode->GetNodeName() == observationNames[i]) {
            OBSERVATION_INDEX = i;
        }
    }
    assert(OBSERVATION_INDEX > -1);

    //Determine the node weights we will loop over
    v_nodeWeights = inputNode->GetWeights_W();

    //For each time series in set
    std::cout << "\tFor node:" << std::endl;
    inputNode->PrintNodeParameters();
    for (int indexTimeSeries = 0; indexTimeSeries < numberOfTimeSeriesInSet; indexTimeSeries++) {
        currentTimeSeries = inputTimeSeriesSet->GetTimeSeriesFromIndex(indexTimeSeries);
        numberOfTimePointsInSeries = currentTimeSeries->GetNumberOfTimePointsInSeries();
        assert(numberOfTimePointsInSeries > 0);
        //For each time point in time series using the first time point as initial conditions
        //We approximate the value of e(t) using e(t-1) + De(t-1).
        for (int indexTimePoint = 1; indexTimePoint < numberOfTimePointsInSeries; indexTimePoint++) {
            currentTimePoint = currentTimeSeries->GetTimePointFromIndex(indexTimePoint);
            previousTimePoint = currentTimeSeries->GetTimePointFromIndex(indexTimePoint - 1);
            //Sum the results of expression * weight
            for (int indexNodeWeightsAndExpression = 0; indexNodeWeightsAndExpression < v_nodeWeights.size(); indexNodeWeightsAndExpression++) {
                observationValue = previousTimePoint->GetObservationFromIndex(indexNodeWeightsAndExpression);
                weightValue = v_nodeWeights[indexNodeWeightsAndExpression];
                summationOfExpressionTimesWeight = summationOfExpressionTimesWeight + (observationValue * weightValue);
                //std::cout << "summationOfExpressionTimesWeight is: " << summationOfExpressionTimesWeight << std::endl;
            }
            intermediateCalculationRateOfChangeOfNode  = inputNode->GetBasalExpression_Beta() + summationOfExpressionTimesWeight;
            //std::cout << "intermediateCalculationRateOfChangeOfNode is: " << intermediateCalculationRateOfChangeOfNode << std::endl;
            intermediateCalculationRateOfChangeOfNode  = this->SigmoidalTransfer(intermediateCalculationRateOfChangeOfNode);
            //std::cout << "intermediateCalculationRateOfChangeOfNode is: " << intermediateCalculationRateOfChangeOfNode << std::endl;
            intermediateCalculationRateOfChangeOfNode  = intermediateCalculationRateOfChangeOfNode - (previousTimePoint->GetObservationFromIndex(OBSERVATION_INDEX) * inputNode->GetDecayRate_Gamma());
            //std::cout << "intermediateCalculationRateOfChangeOfNode is: " << intermediateCalculationRateOfChangeOfNode << std::endl;
            intermediateCalculationRateOfChangeOfNode  = intermediateCalculationRateOfChangeOfNode / inputNode->GetScaleFactor_Tau();
            //std::cout << "intermediateCalculationRateOfChangeOfNode is: " << intermediateCalculationRateOfChangeOfNode << std::endl;

            calculatedExpressionOfNode = intermediateCalculationRateOfChangeOfNode + previousTimePoint->GetObservationFromIndex(OBSERVATION_INDEX);
            observedExpressionOfNode = currentTimePoint->GetObservationFromIndex(OBSERVATION_INDEX);
            std::cout << "Calculated expression is: " << calculatedExpressionOfNode << " and observed is: " << observedExpressionOfNode << "\t";
            differenceBetweenCalcAndObserved = calculatedExpressionOfNode - observedExpressionOfNode;
            squareDifferenceBetweenCalcAndObserved = differenceBetweenCalcAndObserved * differenceBetweenCalcAndObserved;
            std::cout << "difference is: " << differenceBetweenCalcAndObserved << " and sqare is: " << squareDifferenceBetweenCalcAndObserved << "\t";

            meanSquareErrorIntermediate = meanSquareErrorIntermediate + squareDifferenceBetweenCalcAndObserved;
            std::cout << "MSE sum is now: " << meanSquareErrorIntermediate << std::endl;
            //std::cout << "meanSquareErrorIntermediate is: " << meanSquareErrorIntermediate << std::endl;
        }
    }
    meanSquareErrorIntermediate = ( 1.0 / (numberOfTimeSeriesInSet * numberOfTimePointsInSeries) ) * meanSquareErrorIntermediate;
    penaltyTerm = this->CalculatePenaltyTerm(v_nodeWeights);
    //std::cout << "Penalty term is: " << penaltyTerm << std::endl;
    meanSquareErrorIntermediate = meanSquareErrorIntermediate + penaltyTerm;

    //std::cout << "Final meanSquareErrorIntermediate is: " << meanSquareErrorIntermediate << std::endl;

    meanSquareError = meanSquareErrorIntermediate;
    std::cout << "\tmeanSquareError from CalculateNodeFitness: " << meanSquareError << std::endl;
    return meanSquareError;
}

double GeneticAlgorithm::CalculatePenaltyTerm(std::vector<double> inputNodeWeights){
    //sort the weights on ascending order of magnitude
    for (int i = 0; i < inputNodeWeights.size(); i++) {
        if (inputNodeWeights[i] < 0) {
            inputNodeWeights[i] = inputNodeWeights[i] * -1.0;
        }
    }
    std::sort(inputNodeWeights.begin(), inputNodeWeights.end());
    double weightSum_Wij = 0.0;
    double penaltyTerm = 0.0;
    //sum them from 1 to N-I, where N is the number of genes and I the maximum number of allowed regulations.

    int indexToSum = inputNodeWeights.size() - this->MAX_ALLOWED_INTERACTIONS_I;

    if (indexToSum <= 0) {
        penaltyTerm = 0.0;
        return penaltyTerm;
    }
    else {
        for (int i = 0; i < indexToSum; i++) {
            weightSum_Wij = weightSum_Wij + inputNodeWeights[i];
        }
        penaltyTerm = this->PRUNING_CONSTANT_C * weightSum_Wij;
        return penaltyTerm;

    }

}



void GeneticAlgorithm::ConstrainNodeParametersToRanges(NodeParameters_Omega* inputNode) {
    std::string nodeName = inputNode->GetNodeName();
    std::vector<double> v_weights = inputNode->GetWeights_W();
    double beta = inputNode->GetBasalExpression_Beta();
    double gamma = inputNode->GetDecayRate_Gamma();
    double tau = inputNode->GetScaleFactor_Tau();

    for (int i = 0; i < v_weights.size(); i++) {
        if (v_weights[i] < this->WEIGHT_RANGE.first) {
            v_weights[i] = this->WEIGHT_RANGE.first;
        }
        if (v_weights[i] > this->WEIGHT_RANGE.second) {
            v_weights[i] = this->WEIGHT_RANGE.second;
        }
        else {
            //No change necessary.
        }
    }

    if (beta < this->BETA_RANGE.first) {
        beta = this->BETA_RANGE.first;
    }
    if (beta > this->BETA_RANGE.second) {
        beta = this->BETA_RANGE.second;
    }
    else {
        //No change necessary.
    }

    if (gamma < this->GAMMA_RANGE.first) {
        gamma = this->GAMMA_RANGE.first;
    }
    if (gamma > this->GAMMA_RANGE.second) {
        gamma = this->GAMMA_RANGE.second;
    }
    else {
        //No change necessary.
    }

    if (tau < this->TAU_RANGE.first) {
        tau = this->TAU_RANGE.first;
    }
    if (tau > this->TAU_RANGE.second) {
        tau = this->TAU_RANGE.second;
    }
    else {
        //No change necessary.
    }

    inputNode->SetNodeParameters(nodeName, v_weights, beta, gamma, tau);
}

//////////////////////////////////////////////////////////////////////////
//Population

Population::Population() {
    //std::cout << "Empty Population Constructor called" << std::endl;
}

Population::Population(std::vector<NeuralNetwork*> inputNeuralNetworkPopulation) : v_neuralNetworkPopulation(inputNeuralNetworkPopulation) {
    //std::cout << "Population Constructor called with input vector." << std::endl;
    for (int i = 0; i < this->v_neuralNetworkPopulation.size(); i++) {
        this->v_fitnessValues.push_back(-1.0);
    }
}

void Population::PrintPopulation() {
    std::cout << "Printing Population: " << std::endl;
    assert(this->v_neuralNetworkPopulation.size() > 0);
    for (int i = 0; i < this->v_neuralNetworkPopulation.size(); i++) {
        std::cout << "\t\tIndividual " << i << std::endl;
        this->v_neuralNetworkPopulation[i]->PrintNeuralNetwork();
        std::cout << "Fitness Value: " << this->v_fitnessValues[i] << std::endl;
    }
}

NeuralNetwork* Population::GetNetworkFromIndex(int inputNetworkIndex) {
    return this->v_neuralNetworkPopulation[inputNetworkIndex];
}

void Population::ReplaceIndividual(int indexOfIndividualToReplace, NeuralNetwork* inputReplacement) {
    //std::cout << "ReplacingIndividual at index: " << indexOfIndividualToReplace << std::endl;
    delete this->v_neuralNetworkPopulation[indexOfIndividualToReplace];
    this->v_neuralNetworkPopulation[indexOfIndividualToReplace] = inputReplacement;
}

void Population::SetFitnessValue(int indexOfFitnessValue, double fitnessValue) {
    this->v_fitnessValues[indexOfFitnessValue] = fitnessValue;
}

int Population::GetPopulationSize() {
    return this->v_neuralNetworkPopulation.size();
}

NeuralNetwork* Population::GetMostFitIndividual() {
    auto minimum = std::min_element(std::begin(this->v_fitnessValues), std::end(this->v_fitnessValues));
    auto minIndex = std::distance(std::begin(this->v_fitnessValues), minimum);
    //std::cout << "Index of most fit individual is " << minIndex << std::endl;
    return this->GetNetworkFromIndex(minIndex);
}

double Population::GetFitnessOfMostFitIndividual() {
    auto minimum = std::min_element(std::begin(this->v_fitnessValues), std::end(this->v_fitnessValues));
    auto minIndex = std::distance(std::begin(this->v_fitnessValues), minimum);
    //std::cout << "Index of most fit individual is " << minIndex << std::endl;
    return this->v_fitnessValues[minIndex];
}
double Population::GetFitnessOfLeastFitIndividual() {
    auto maximum = std::max_element(std::begin(this->v_fitnessValues), std::end(this->v_fitnessValues));
    auto maxIndex = std::distance(std::begin(this->v_fitnessValues), maximum);
    //std::cout << "Index of least fit individual is " << maxIndex << std::endl;
    return this->v_fitnessValues[maxIndex];
}
