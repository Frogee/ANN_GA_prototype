#ifndef DATAUTIL_H
#define DATAUTIL_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

class ObservationSet {
	public:
        ObservationSet();
		ObservationSet(std::vector<double> v_inputObservations);
		double GetObservationFromIndex(int observationIndex);
		void PrintObservationSet();
	private:
		std::vector<double> v_observations;
};

class TimePoint {
	public:
        TimePoint();
		TimePoint(ObservationSet* inputObservationSet);
		double GetObservationFromIndex(int observationIndex);
		void PrintTimePoint();
	private:
		ObservationSet* observationSet;
};

class TimeSeries {
	public:
        TimeSeries();
		TimeSeries(std::vector<TimePoint*> v_inputTimePoints);
		TimePoint* GetTimePointFromIndex(int timePointIndex);
		void SetObservationNames(std::vector<std::string> inputObservationNames);
		std::vector<std::string> GetObservationNames();
        int GetNumberOfTimePointsInSeries();
		void PrintObservationNames();
		void PrintTimeSeries();
	private:
		std::vector<TimePoint*> v_timePoints;
		std::vector<std::string> v_observationNames;
};

class TimeSeriesSet {
	public:
        TimeSeriesSet();
		TimeSeriesSet(std::vector<TimeSeries*> v_timeSeries);
		TimeSeries* GetTimeSeriesFromIndex(int timeSeriesIndex);
		double GetObservationFromCoordinates(int timeSeriesIndex, int timePointIndex, int observationIndex);
		std::vector<std::string> GetObservationNames();
		int GetNumberOfTimeSeriesInSet();
		void AddTimeSeriesToSet(TimeSeries * inputTimeSeries);
		void PrintTimeSeriesSet();
	private:
		std::vector<TimeSeries*> v_timeSeries;
};

class TimeSeriesSetDataFile {
    public:
        TimeSeriesSetDataFile(std::ifstream * fileStream);
        void PrintDataFile();
        std::string GetLine();
        bool Good();
    private:
        std::ifstream * dataFileStream;
};

class TimeSeriesSetDataFileParser {
    public:
        TimeSeriesSetDataFileParser();
        TimeSeriesSet* ParseDataFile(TimeSeriesSetDataFile * inputTimeSeriesSetDataFile);
    private:
        std::vector<std::string> SplitString(const std::string &inputString, char delimeter);   //Taken from http://stackoverflow.com/questions/236129/how-to-split-a-string-in-c
        std::vector<std::string> & SplitString(const std::string &inputString, char delimeter, std::vector<std::string> &elems);
};


#endif
