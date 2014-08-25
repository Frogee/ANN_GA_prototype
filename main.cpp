#include <iostream>
#include <fstream>
#include <cassert>
#include "dataUtil.h"
#include "neuralNetwork.h"
#include "geneticAlgorithm.h"

///This is an implementation inspired by Noman, Palafox, Iba "Reconstruction of Gene Regulatory Networks from Gene Expression Data Using Decoupled Recurrent Neural Network Model
///   Unlike Noman et al, this solves for the whole system of equations for the network when determining fitness rather than treating it as subproblems.
///   Treating it as subproblems led to poorly performing models the way I had implemented it.
///   As such, a fair bit of code still remains from when the Genetic Algorithm operated on nodes rather than networks that needs to be cleaned up.
///
///Currently the network size has to be defined for the preprocessor in geneticAlgorithm.cpp and then compiled.
///   I don't know how to get it to work at runtime due to odeint's requirements for the system of equations
///   TODO: See if this can be solved so that the system of equations can be defined at runtime based on input data.
///
///TODO: Take an input data file from the command line.
///


int main()
{
    //Get a pointer to the initialized TimeSeriesSet containing all the data.
    TimeSeriesSet * timeSeriesSet;
    const char * fileName = "/home/workgolem_rfm/Documents/Sorghum_Genomics/Development/artificial_neural_network/prototype/prototype1/RNN_prototype1/RNN_prototyping1/bin/Debug/RNN_simulated_data_Debug.csv";
    std::ifstream inFileStream(fileName, std::ifstream::in);
    assert(inFileStream.is_open());
    TimeSeriesSetDataFile * timeSeriesSetDataFile = new TimeSeriesSetDataFile(&inFileStream);
    TimeSeriesSetDataFileParser * timeSeriesSetDataFileParser = new TimeSeriesSetDataFileParser();
    timeSeriesSet = timeSeriesSetDataFileParser->ParseDataFile(timeSeriesSetDataFile);

    timeSeriesSet->PrintTimeSeriesSet();

    //Initialize a Neural Network based on the Time Series Data.
    NeuralNetwork * neuralNetwork = new NeuralNetwork;
    NeuralNetworkInitializer * neuralNetworkInitializer = new NeuralNetworkInitializer;
    neuralNetworkInitializer->InitializeNeuralNetwork(neuralNetwork, timeSeriesSet);

    neuralNetwork->PrintNeuralNetwork();

    //Evolve the Neural Network based on the Time Series Data.
    GeneticAlgorithm * geneticAlgorithm = new GeneticAlgorithm;

    geneticAlgorithm->EvolveNetwork(neuralNetwork, timeSeriesSet);

    std::cout << "After network evolution:" << std::endl;
    neuralNetwork->PrintNeuralNetwork();

    geneticAlgorithm->CalculateNetworkFitnessVerbose(neuralNetwork, timeSeriesSet);

    delete timeSeriesSetDataFile;
    delete neuralNetwork;
    delete neuralNetworkInitializer;
    return 0;
}
