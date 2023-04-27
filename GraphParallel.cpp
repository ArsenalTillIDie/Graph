// Graph.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


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

#include "GraphClass.h"



const int UNREACHABLE = -1;



Graph generateGraph(int nVertices, double chance) {
	const static int maxWeight = 1;
	/*
	weights = new int* [nVertices];
	for (int i = 0; i < nVertices; i++) {
		weights[i] = new int[nVertices];
		for (int j = 0; j < nVertices; j++)
			weights[i][j] = 0;
	}*/
	Graph graph(nVertices);
	for (int i = 0; i < nVertices; i++) {
		for (int j = i + 1; j < nVertices; j++) {
			int rN = rand();
			if (rN < RAND_MAX * chance) {
				graph.addEdge(i, j, 1);
			}
		}
	}
	return graph;
}

Graph washington(int N, int**& weights) {
	const int nVertices = 3 * N + 3;
	/*
	weights = new int* [nVertices];
	for (int i = 0; i < nVertices; i++) {
		weights[i] = new int[nVertices];
		for (int j = 0; j < nVertices; j++)
			weights[i][j] = 0;
	}*/

	Graph graph(nVertices);
	//for (int i = 0; i < 3 * N + 3; i++) {
		//Vertex vx(Pair(i, N), {}, {});
		//graph.vertices[i] = vx;
	//}
	graph.addEdge(0, 1, N);
	int currEdge = 3;
	for (int i = 0; i < N; i++) {
		graph.addEdge(1, currEdge, N);
		currEdge++;
		graph.addEdge(currEdge - 1, currEdge, N);
		currEdge++;
		graph.addEdge(currEdge - 1, 2, N);
	}
	graph.addEdge(2, currEdge, N);
	currEdge++;
	for (int i = 0; i < N - 1; i++) {
		graph.addEdge(currEdge - 1, currEdge, N);
		currEdge++;
	}
	return graph;
}

/*
template<class T> Graph<int> graphFromPoints(std::vector<T> pts, int**& weights, double(*distance)(T, T)) {
	const int nVertices = pts.size();
	std::vector<Vertex<int>> vertices(nVertices);
	weights = new int* [nVertices];
	for (int i = 0; i < nVertices; i++) {
		weights[i] = new int[nVertices];
		for (int j = 0; j < nVertices; j++)
			weights[i][j] = 0;
	}
	Graph<int> graph(vertices, weights);
	for(int i = 0; i < nVertices; i++)
		for (int j = 0; j < nVertices; j++) {
			graph.addEdge(i, j, int(distance(pts[i], pts[j]) * 100));
		}
	return graph;
}
*/

Graph graphFromEdges(int nVertices, std::vector<std::vector<int>> edges) {
	Graph graph(nVertices);
	for (int i = 0; i < edges.size(); i++) {
		graph.addEdge(edges[i][0], edges[i][1], 1);
	}
	return graph;
}

/**
Graph<int> graphFromMM(std::string path) {
	std::fstream fs(path);
	int v1, v2, weight;
	std::string line;
	do {
		fs >> line;
	} while (line[0] == '%');
}
*/
std::vector<std::vector<int>> edges = { {0,1}, {0,2}, {6,2}, {1,5}, {2,3}, {2,4}, {1,3}, {1,6} };

std::string stageTimeInfo(std::vector<double> times, double full, int index) {
	std::stringstream sstr;
	sstr << times[index] << " (" << times[index] / full * 100 << "%)";
	return sstr.str();
}


enum Mode { Daniel = 0, Brandes, compare };

Environment env;

int main(int argc, char** argv)
{
	//srand(time(NULL));
	//srand(11514);
	srand(1114);
	//srand(time(NULL));
	int n, nThreads; Mode mode = compare;
	if (argc < 2) {
		throw("Graph size unspecified");
	}
	n = std::atoi(argv[1]);
	if (argc >= 3) {
		mode = (Mode)(std::atoi(argv[2]));
	}
	if (argc >= 4) {
		env.nThreads = (Mode)(std::atoi(argv[3]));
	}
	else env.nThreads = omp_get_max_threads();

	std::cout << std::endl << "n = " << n << std::endl;
	//Graph graph = graphFromEdges(7, edges);
	Graph graph = generateGraph(n, 3.0 / n);

	std::vector<int> comp; int nComp;
	graph.DFS(0, &comp, &nComp);
	//sum = 0;
	std::vector<bool> componentProcessed(nComp);
	componentProcessed[0] = true;
	for (int i = 0; i < n; i++) {
		if (componentProcessed[comp[i]]) continue;
		int rV;
		do {
			rV = rand() % n;
		} while (comp[rV] != comp[i] - 1);
		graph.addEdge(i, rV, 1);
		componentProcessed[comp[i]] = true;

	}


	omp_set_num_threads(env.nThreads);

	//std::cout << omp_get_num_threads() << " threads\n";
	std::cout << "NZ = " << graph.adjacencies.size() << std::endl;
	std::vector<double> bcsf;
	double* bcs = new double[graph.size()];
	double begin_time;
	if (mode == Daniel || mode == compare) {
		std::vector<double> stageTimes(6);
		begin_time = omp_get_wtime();
		bcsf = graph.fastBC(stageTimes);
		double fastBCFull = omp_get_wtime() - begin_time;
		std::cout << "Daniel: " << fastBCFull << std::endl << std::endl;
		std::cout << "That is, clustering: " << stageTimeInfo(stageTimes, fastBCFull, 0) << "," << std::endl;
		std::cout << "finding border nodes: " << stageTimeInfo(stageTimes, fastBCFull, 1) << "," << std::endl;
		std::cout << "building the HSN: " << stageTimeInfo(stageTimes, fastBCFull, 2) << "," << std::endl;
		std::cout << "finding external nodes: " << stageTimeInfo(stageTimes, fastBCFull, 3) << "," << std::endl;
		std::cout << "computing local dependency scores and finding equivalence classes: " << stageTimeInfo(stageTimes, fastBCFull, 4) << "," << std::endl;
		std::cout << "computing global dependency scores and betweenness centralities: " << stageTimeInfo(stageTimes, fastBCFull, 5) << "." << std::endl << std::endl;
	}
	if (mode == Brandes || mode == compare) {
		//for(double alpha = 0.1; alpha < 1; alpha += 0.1)
			//for (double beta = 0.1; beta < 1; beta += 0.1) {
				begin_time = omp_get_wtime();
				graph.brandes(bcs, 0.1, 0.1);
				std::cout <</* "Alpha = " << alpha << ", beta = "<< beta <<*/"Brandes: " << float(omp_get_wtime() - begin_time) << std::endl;
			//}
	}
	if (mode == compare) {
		double maxDiff = -1; int idx = -1; double bcsidx = -1, bcsnidx = -1;

		//<< " " << bcsn[i] << std::endl;
		for (int i = 0; i < bcsf.size(); i++)

			if (abs(bcs[i] - bcsf[i]) > maxDiff) {
				//std::cout << "i = " << i << ", Brandes BC is " << bcs[i] << ", naive BC is " << bcsf[i] << std::endl;
				idx = i;
				bcsidx = bcs[i];
				bcsnidx = bcsf[i];
				maxDiff = abs(bcs[i] - bcsf[i]);
			}
		std::cout << "Biggest difference is " << abs(bcsidx - bcsnidx) << " at i = " << idx << ", Brandes BC is " << bcsidx << ", Daniel BC is " << bcsnidx << std::endl;
	}
	//}


}

