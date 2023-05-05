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

struct Point {
	double x;
	double y;
	Point(double _x, double _y) {
		x = _x;
		y = _y;
	}
};

double distance(Point a, Point b) {
	return pow((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y), 0.5);
}

struct Iris {
	double sepalLength;
	double sepalWidth;
	double petalLength;
	double petalWidth;
};

double irisDistance(Iris a, Iris b) {
	return pow(pow(a.sepalLength - b.sepalLength, 2) + pow(a.sepalWidth - b.sepalWidth, 2) +
		pow(a.petalLength - b.petalLength, 2) + pow(a.petalWidth - b.petalWidth, 2), 0.5);
}

std::vector<Iris> readIrisesFromFile(std::string fileName) {
	std::ifstream file;
	std::vector<Iris> irises;
	file.open(fileName);
	int i = 0;
	std::string line;
	while (std::getline(file, line)) {
		if (line == "") break;
		std::istringstream iss(line);
		std::string substr;
		Iris iris;
		std::getline(iss, substr, ',');
		iris.sepalLength = std::stod(substr);
		std::getline(iss, substr, ',');
		iris.sepalWidth = std::stod(substr);
		std::getline(iss, substr, ',');
		iris.petalLength = std::stod(substr);
		std::getline(iss, substr, ',');
		iris.petalWidth = std::stod(substr);
		//iss >> iris.sepalLength >> iris.sepalWidth >> iris.petalLength >> iris.petalWidth;
		irises.push_back(iris);
	}
	file.close();
	return irises;
}
/*
template<class T> class Node {
public:
	T data;
	Node* next;
	Node(T val, Node* nxt = nullptr) {
		data = val;
		next = nxt;
	}
};
*/
/*
template<class T> class List {
private:
	Node<T>* first;
public:
	List() {
		first = nullptr;
	}
	List(const List& lst) : List() {
		Node<T>* current = nullptr;
		for (iterator it = lst.begin(); it != end(); ++it) {
			current = insert_after(it->data, current);
		}
	}
	~List() {
		iterator it = begin();
		while (it != end()) {
			iterator nit(it->next);
			delete* it;
			it = nit;
		}
	}
	List<T>& operator=(const List<T> lst) {
		first = nullptr;
		Node<T>* current = nullptr;
		for (iterator it = lst.begin(); it != end(); ++it) {
			current = insert_after(it->data, current);
		}
		return *this;
	}
	Node<T>* getFirst() {
		return first;
	}
	T front() {
		return first->data;
	}
	bool empty() {
		return first == nullptr;
	}
	void clear() {
		iterator it = begin();
		while (it != end()) {
			iterator nit(it->next);
			delete* it;
			it = nit;
		}
		first = nullptr;
	}
	Node<T>* insert_after(T data, Node<T>* n) {
		if (n == nullptr) return push_front(data);
		Node<T>* pnn = new Node<T>(data, n->next);
		n->next = pnn;
		return pnn;
	}
	void erase_after(Node<T>* n) {
		if (n == nullptr) {
			pop_front();
			return;
		}
		Node<T>* rem = n->next;
		n->next = rem->next;
		delete rem;
	}
	Node<T>* push_front(T data) {
		Node<T>* pnn = new Node<T>(data, first);
		first = pnn;
		return pnn;
	}
	void pop_front() {
		Node<T>* rem = first;
		first = rem->next;
		delete rem;
	}
	class iterator {
	private:
		Node<T>* current;
	public:
		iterator() {
			current = first;
		}
		iterator(Node<T>* pnode) {
			current = pnode;
		}
		iterator(const iterator& it) {
			current = it.current;
		}
		Node<T>*& operator*() {
			return current;
		}
		Node<T>* operator->() {
			return current;
		}
		iterator operator++() {
			current = current->next;
			return current;
		}
		bool operator==(iterator it) {
			return it.current == current;
		}
		bool operator!=(iterator it) {
			return !(it == *this);
		}
	};
	iterator begin() const {
		iterator it(first);
		return it;
	}
	iterator end() const {
		iterator it(nullptr);
		return it;
	}
};
*/




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

Graph generateConnectedGraph(int n, double avgDeg, int seed = -1){
	static int called = 1;
	called++;
	if(seed >= 0) srand(seed);
	Graph graph = generateGraph(n, avgDeg / n);

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
	srand(time(NULL) + called);
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


enum Mode { Daniel = 0, Brandes, compare, test };

Environment env;

int main(int argc, char** argv)
{
	//srand(time(NULL));
	//srand(11514);
	//srand(111514);
	//srand(time(NULL));
	int n, nThreads; Mode mode = compare; int hybrid = 0; double alpha = 0.1; double beta = 0.1; double avgDeg = 3.0;
	int seed = 111514;
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
	if (argc >= 5) {
		hybrid = (bool)(std::atoi(argv[4]));
	}
	if (argc >= 6) {
		alpha = std::stod(argv[5]);
	}
	if (argc >= 7) {
		beta = std::stod(argv[6]);
	}
	if (argc >= 8) {
		avgDeg = std::stod(argv[7]);
	}
	if(argc >= 9){
		seed = std::atoi(argv[8]);
	}
	srand(seed);
	std::cout << std::endl << "n = " << n << std::endl;
	//Graph graph = graphFromEdges(7, edges);
	Graph graph = generateGraph(n, avgDeg / n);

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
		bcsf = graph.fastBC(stageTimes, hybrid, alpha, beta);
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
				graph.brandes(bcs, hybrid, alpha, beta);
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
	if (mode == test) {
		std::ofstream file;
		file.open("Hybrid Brandes times.txt");
		for(double dens = 3; dens < 30; dens *= 2){
			Graph graph = generateConnectedGraph(4000, dens, seed);
			for(int thr = 1; thr <= 1; thr *= 2){
				graph.setNThreads(thr);
				for(double alpha = 0.1; alpha < 1; alpha += 0.2){
					for(double beta = 0.1; beta < 1; beta += 0.2){
						begin_time = omp_get_wtime();
						graph.brandes(bcs, hybrid, alpha, beta);
						float time = float(omp_get_wtime() - begin_time);
						std::cout << "Alpha = " << alpha << ", beta = "<< beta << ", " << thr << " threads. Brandes: " << float(omp_get_wtime() - begin_time) << std::endl;
						file << time << " ";
					}
				}
				file << "\n";
			}
				 
		}
		file.close();
	}
	//}


}

