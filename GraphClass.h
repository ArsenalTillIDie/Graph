#include <iostream>
#include <vector>
#include <list>
#include <queue>
#include <set>
#include <ctime>
#include <cstdlib>
#include <bitset>
#include <ratio>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <stack>
#include <fstream>
#include <deque>
#include <omp.h>
//#include "Vertex.h"
#pragma once

struct Environment {
	int nThreads;
	Environment() {}
	Environment(int n) {
		nThreads = n;
	}
};

struct EdgeInfo {
	int a;
	int b;
	int weight;
	bool operator<(const EdgeInfo ei) {
		if (weight < ei.weight)
			return true;
		else return false;
	}
};

struct VertexDistance {
	int vertex;
	int distance;
	VertexDistance() {};
	VertexDistance(int pvx, int d);
	bool operator<(const VertexDistance vd) const;
	VertexDistance(const VertexDistance& vd);
};

class Graph {
	int nThreads;
protected:
	void convertQueueToBitset(std::queue<int> q, std::vector<bool>& frontier);
	void convertBitsetToQueue(std::vector<bool> frontier, std::queue<int>& q);
	void BFS(int current, std::vector<bool>& markedVertices,
		std::vector<int>& path, std::vector<int>& distances, std::vector<int>& predecessors, std::queue<int>& q,
		double a);
	void BFSBottomUp(int current, std::vector<bool>& markedVertices,
		std::vector<int>& path, std::vector<int>& distances, std::vector<int>& predecessors, std::vector<bool>& frontier,
		double b);
	void collectPaths(int source, int dest, std::vector<std::vector<int>>& paths, std::vector<int>* predecessors);
	void brandesBFS(int startingVertex, std::vector<int> clusters, std::vector<std::vector<int>>& predecessors,
		std::vector<double>& deltas, std::vector<int>& sigmas, std::vector<double>& cbs/*, std::vector<double>* localDeltas = nullptr*/);
	void modifiedBrandesV1BFS(int startingVertex, std::vector<int>& clusters, std::vector<int>* predecessors,
		double* deltas, int* sigmas, double* localDeltas, int* localSigmas, int* distances,
		std::vector<bool>& externalNodes);
	void modifiedBrandesV2BFS(int startingVertex, std::vector<int>& clusters, std::vector<int>* predecessors,
		double* globalDeltas, int nClusters);
	void findBorderNodes(std::vector<int>& clusters, std::vector<bool>&);
	void localBFSAllPaths(int startingVertex, std::vector<int>& clusters, std::vector<int>* predecessors);
	void buildHSN(std::vector<int>& clusters, std::vector<bool>& borderNodes, std::vector<bool>&);
	void findExternalNodes(std::vector<bool>& hsn, std::vector<int>& clusters, std::vector<bool>& borderNodes,
		std::vector<std::vector<bool>>& updatedClusters, int nClusters, std::vector<bool>*);
	struct equivalenceClass {
		std::vector<int> indices;
		std::vector<int> distances;
		std::vector<double> sigmas;
		equivalenceClass(int vertex, std::vector<int> _distances, std::vector<double> _sigmas);
		void addVertex(int vertex);
		int random();
	};
	void localDeltas(std::vector<int>& clusters, std::vector<bool>& borderNodes, std::vector<std::vector<bool>>& updatedClusters, std::vector<bool>* externalNodes,
		double* localDeltas, std::vector<std::vector<int>>& clusterVector, std::vector<std::vector<equivalenceClass> >& classes, int nClusters,
		int maxCluster);

	std::vector<typename Graph::equivalenceClass> findClasses(int** distances, double** sigmas,
		std::vector<int>& vectorCluster, std::vector<bool>& borderNodes);
	void globalDeltas(std::vector<std::vector<equivalenceClass>>& classes, std::vector<int>& clusters, int nClusters, double*);
	void DFS(int current, std::vector<bool>& markedVertices, std::vector<int>& path, std::vector<int>& components, int currentComponent);
	void dijkstra(int current, std::vector<bool>& markedVertices, std::vector<int>& distances, std::vector<int>& predecessors, std::set<VertexDistance>& s);
	void initDSU(std::vector<int>& p, std::vector<int>& rk);
	int getRoot(std::vector<int>& p, std::vector<int>& rk, int v);
	bool merge(std::vector<int>& p, std::vector<int>& rk, int a, int b);
public:
	std::vector<int> adjacencies;
	std::vector<int> weights;
	std::vector<int> row_index;
	Graph() {};
	Graph(int size);
	Graph(const Graph& g);
	~Graph() {};
	Graph operator=(const Graph& g);
	int size() {
		return row_index.size() - 1;
	}
	int weight(int v1, int v2);
	std::vector<int> BFS(int startingVertex, std::vector<int>* d = nullptr, std::vector<int>* pred = nullptr, double a = 1, double b = 1);
	void BFSAllPaths(int startingVertex, std::vector<std::vector<std::vector<int>>>& _paths);
	std::vector<int> DFS(int startingVertex = 0, std::vector<int>* comp = nullptr, int* nComponents = nullptr);
	void dijkstra(int startingVertex, std::vector<int>* d = nullptr, std::vector<int>* pred = nullptr);
	Graph prim(int startingVertex = 0);
	Graph kruskal();
	int heaviestEdge(int& v1, int& v2, int& idx);
	std::vector<int> clusterise(int nClusters, int algorithm = 0);
	int sumAllWeights(std::vector<int> clusters, int cluster = -1, int vertex = -1, int* incC = nullptr, int* incV = nullptr, int* between = nullptr);
	std::vector<int> louvain(int& nClusters, int desiredNClusters = 0);
	void adoptSingletons(std::vector<int>& clusters, int& nClusters);
	std::vector<double> brandes(std::vector<int> clusters = {});
	std::vector<double> brandesNaive();
	std::vector<double> fastBC();
	void addEdge(int v1, int v2, int weight = 0);
	void assignWeight(int v1, int v2, int weight);
	bool areNeighbours(int v1, int v2);
	void removeEdge(int v1, int v2);
};
