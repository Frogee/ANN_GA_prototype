#include "dataUtil.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include <stdlib.h>

//////////////////////////////////////////////////////////////////////////
//TimeSeriesSetDataFile
TimeSeriesSetDataFile::TimeSeriesSetDataFile(std::ifstream * fileStream) : dataFileStream(fileStream) {
    std::cout << "TimeSeriesSetDataFile constructed with data file." << std::endl;
}

void TimeSeriesSetDataFile::PrintDataFile() {
    char c = this->dataFileStream->get();
    while (this->dataFileStream->good()) {
        std::cout << c;
        c = this->dataFileStream->get();
    }
    std::cout << "Set the data file here" << std::endl;
}

std::string TimeSeriesSetDataFile::GetLine() {
    std::string line;
    assert(this->dataFileStream->good());
    std::getline((*this->dataFileStream), line);
    return line;
}

bool TimeSeriesSetDataFile::Good() {
    return this->dataFileStream->good();
}
//////////////////////////////////////////////////////////////////////////
//TimeSeriesSet
//TimeSeriesSet Constructor
TimeSeriesSet::TimeSeriesSet() {
    std::cout << "Empty Time Series Set constructor called" << std::endl;
}

TimeSeriesSet::TimeSeriesSet(std::vector<TimeSeries*> v_inputSeries) {
	this->v_timeSeries = v_inputSeries;
}


TimeSeries* TimeSeriesSet::GetTimeSeriesFromIndex(int timeSeriesIndex) {
	return this->v_timeSeries[timeSeriesIndex];
}

double TimeSeriesSet::GetObservationFromCoordinates(int timeSeriesIndex, int timePointIndex, int observationIndex) {
	TimeSeries * timeSeries = this->GetTimeSeriesFromIndex(timeSeriesIndex);
	TimePoint * timePoint = timeSeries->GetTimePointFromIndex(timePointIndex);
	double observation = timePoint->GetObservationFromIndex(observationIndex);
	return observation;
}

int TimeSeriesSet::GetNumberOfTimeSeriesInSet() {
    return this->v_timeSeries.size();
}

std::vector<std::string> TimeSeriesSet::GetObservationNames() {
    assert(this->v_timeSeries.size() > 0);
    return this->v_timeSeries[0]->GetObservationNames();
}

void TimeSeriesSet::AddTimeSeriesToSet(TimeSeries * inputTimeSeries) {
    this->v_timeSeries.push_back(inputTimeSeries);
}

void TimeSeriesSet::PrintTimeSeriesSet() {
    std::cout << "Printing the Time Series Set:" << std::endl;
    for (int i = 0; i < this->v_timeSeries.size(); i++) {
        std::cout << "Time series " << i+1 << ":" << std::endl;
        this->v_timeSeries[i]->PrintTimeSeries();
    }
}

//////////////////////////////////////////////////////////////////////////
//TimeSeries
TimeSeries::TimeSeries() {
    std::cout << "Empty Time Series constructor" << std::endl;
}

TimeSeries::TimeSeries(std::vector<TimePoint*> v_inputTimePoints) {
    this->v_timePoints = v_inputTimePoints;
}

TimePoint* TimeSeries::GetTimePointFromIndex(int timePointIndex) {
    return this->v_timePoints[timePointIndex];
}

void TimeSeries::SetObservationNames(std::vector<std::string> inputObservationNames) {
    for (int i = 0; i < inputObservationNames.size(); i++) {
        this->v_observationNames.push_back(inputObservationNames[i]);
    }
}

std::vector<std::string> TimeSeries::GetObservationNames() {
    return this->v_observationNames;
}

int TimeSeries::GetNumberOfTimePointsInSeries() {
    return this->v_timePoints.size();
}

void TimeSeries::PrintObservationNames() {
    for (int i = 0; i < this->v_observationNames.size(); i++) {
        std::cout << this->v_observationNames[i] << "\t";
    }
    std::cout << std::endl;
}

void TimeSeries::PrintTimeSeries() {
    this->PrintObservationNames();
    for (int i = 0; i < this->v_timePoints.size(); i++) {
        this->v_timePoints[i]->PrintTimePoint();
    }
}

//////////////////////////////////////////////////////////////////////////
//TimePoint
TimePoint::TimePoint() {
    std::cout << "Empty TimePoint constructor called" << std::endl;
}

TimePoint::TimePoint(ObservationSet* inputObservationSet) {
    this->observationSet = inputObservationSet;
}

double TimePoint::GetObservationFromIndex(int observationIndex) {
    return this->observationSet->GetObservationFromIndex(observationIndex);
}

void TimePoint::PrintTimePoint() {
    this->observationSet->PrintObservationSet();
}


//////////////////////////////////////////////////////////////////////////
//ObservationSet
ObservationSet::ObservationSet() {
    std::cout << "Empty ObservationSet constructor called" << std::endl;
}

ObservationSet::ObservationSet(std::vector<double> v_inputObservations) {
    this->v_observations = v_inputObservations;
}

double ObservationSet::GetObservationFromIndex(int observationIndex) {
    return this->v_observations[observationIndex];
}

void ObservationSet::PrintObservationSet() {
    for (int i = 0; i < this->v_observations.size(); i++) {
        std::cout << this->v_observations[i] << "\t";
    }
    std::cout << std::endl;
}


//////////////////////////////////////////////////////////////////////////
//TimeSeriesSetDataFileParser
TimeSeriesSetDataFileParser::TimeSeriesSetDataFileParser() {
    std::cout << "Empty TimeSeriesSetDataFileParser Constructor called" << std::endl;
}



std::vector<std::string> & TimeSeriesSetDataFileParser::SplitString(const std::string &inputString, char delimiter, std::vector<std::string> &elements) {
    std::stringstream sstream(inputString);  //Taken from http://stackoverflow.com/questions/236129/how-to-split-a-string-in-c
    std::string element;
    while (std::getline(sstream, element, delimiter)) {
        elements.push_back(element);
    }
    return elements;
}

std::vector<std::string> TimeSeriesSetDataFileParser::SplitString(const std::string &inputString, char delimiter) {
            std::vector<std::string> elements;   //Taken from http://stackoverflow.com/questions/236129/how-to-split-a-string-in-c
            this->SplitString(inputString, delimiter, elements);
            return elements;
}

TimeSeriesSet* TimeSeriesSetDataFileParser::ParseDataFile(TimeSeriesSetDataFile * inputTimeSeriesSetDataFile) {
    std::cout << "Parsing input data file" << std::endl;
    std::string line;
    std::vector<std::vector<std::string> > vvstr_data;
    std::vector<std::string> vstr_splitLine;

    while (inputTimeSeriesSetDataFile->Good()) {
        line = inputTimeSeriesSetDataFile->GetLine();
        if(inputTimeSeriesSetDataFile->Good()) {
            vstr_splitLine = this->SplitString(line, '\t');
            vvstr_data.push_back(vstr_splitLine);   //For now we just read it into a 2 dimensional vector of strings.
        }
    }

    //Here we parse the 2 dimensonal vector of strings in an appaling manner
    ///TODO: Find a better way to do this.
    TimeSeriesSet * timeSeriesSet = new TimeSeriesSet;
    for (int i = 0; i < vvstr_data.size(); i = i ) {
        assert(vvstr_data[i].size() > 0);
        if (vvstr_data[i][0].find("TimeSeries") != std::string::npos) {   //Found a new time series
            std::cout << "Found a New Time Series" << std::endl;

            std::vector<std::string> observationNames;   //This vector is used further below.
            for (int obsNum = 1; obsNum < vvstr_data[i].size(); obsNum++) {
                observationNames.push_back(vvstr_data[i][obsNum]);
            }

            std::vector<TimePoint*> v_timePoints;
            i++;
            while (vvstr_data[i][0].find("TimeSeries") == std::string::npos) {  //While still reading for the current time series
                std::vector<double> observations;
                for (int obsNum = 1; obsNum < vvstr_data[i].size(); obsNum++) {
                    std::string str_value;
                    double d_value;
                    str_value = vvstr_data[i][obsNum];
                    d_value = atof(str_value.c_str());
                    observations.push_back(d_value);
                }

                ObservationSet * observationSet = new ObservationSet(observations);
                TimePoint * timePoint = new TimePoint(observationSet);
                v_timePoints.push_back(timePoint);
                i++;
                if (i >= vvstr_data.size()) {   //Test to not go outside of vector bounds.
                    break;
                }
            }
            std::cout << "Creating a new time series" << std::endl;
            TimeSeries * timeSeries = new TimeSeries(v_timePoints);
            timeSeries->SetObservationNames(observationNames);
            timeSeriesSet->AddTimeSeriesToSet(timeSeries);
        }
    }
    return timeSeriesSet;
}



