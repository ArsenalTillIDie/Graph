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

template<class T> class Node {
public:
	T data;
	Node* next;
	Node(T val, Node* nxt = nullptr) {
		data = val;
		next = nxt;
	}
};

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
		if (n == nullptr) pop_front();
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

template<class T> struct Pair {
	unsigned int first;
	T second;
	Pair() {

	}
	Pair(unsigned int n1, T n2) {
		first = n1;
		second = n2;
	}
	Pair(const Pair<T>& p) {
		first = p.first;
		second = p.second;
	}
	Pair& operator=(const Pair p) {
		first = p.first;
		second = p.second;
		return *this;
	}
};


template<class T> struct Vertex {
	Pair<T> data;
	List<Vertex<T>*> adjacencyList;
	Vertex() {

	}
	Vertex(Pair<T> d) {
		data = d;
	}
	Vertex(Pair<T> d, List<Vertex<T>*> al) {
		data = d;
		adjacencyList = al;
	}
	Vertex(const Vertex<T>& vx) {
		data = vx.data;
		adjacencyList = vx.adjacencyList;
	}
	Vertex<T>& operator=(const Vertex<T> vx) {
		data = vx.data;
		adjacencyList = vx.adjacencyList;
		return *this;
	}
};

template <class T> struct VertexDistance {
	Vertex<T>* vertex;
	int distance;
	VertexDistance() {

	}
	VertexDistance(Vertex<T>* pvx, int d) {
		vertex = pvx;
		distance = d;
	}
	bool operator<(const VertexDistance<T> vd) const {
		if (distance < vd.distance || (distance == vd.distance && vertex->data.first < vd.vertex->data.first))
			return true;
		else return false;
	}
	VertexDistance(const VertexDistance& vd) {
		vertex = vd.vertex;
		distance = vd.distance;
	}
};

const int UNREACHABLE = -1;

template<class T> class Graph {
protected:
	void convertQueueToBitset(std::queue<Vertex<T>*> q, std::vector<bool>& frontier) {
		std::fill(frontier.begin(), frontier.end(), false);
		while (!q.empty()) {
			frontier[index(q.front())] = true;
			q.pop();
		}
	}
	void convertBitsetToQueue(std::vector<bool> frontier, std::queue<Vertex<T>*>& q) {
		while (!q.empty()) {
			q.pop();
		}
		for (int i = 0; i < frontier.size(); i++)
			if (frontier[i])
				q.push(&(vertices[i]));
	}
	void BFS(Vertex<T>* current, std::vector<bool>& markedVertices,
		std::vector<Vertex<T>*>& path, std::vector<int>& distances, std::vector<Vertex<T>*>& predecessors, std::queue<Vertex<T>*>& q,
		double a) {
		bool iterationComplete = false;
		while (true) {
			Vertex<T>* u = q.front();
			q.pop();
			for (typename List<Vertex<T>*>::iterator it = u->adjacencyList.begin(); it != u->adjacencyList.end(); ++it) {
				if (!(markedVertices[index((*it)->data)])) {
					markedVertices[index((*it)->data)] = true;
					path.push_back((*it)->data);
					distances[index((*it)->data)] = distances[index(u)] + 1;
					predecessors[index((*it)->data)] = u;
					q.push((*it)->data);
				}
			}
			if (!q.empty() && distances[index(u)] + 1 == distances[index(q.front())]) {
				std::queue<Vertex<T>*> q2 = q;
				int mf = 0, mu = 0;
				while (!q2.empty()) {
					Vertex<T>* vx = q2.front();
					q2.pop();
					for (typename List<Vertex<T>*>::iterator it = vx->adjacencyList.begin(); it != vx->adjacencyList.end(); ++it) {
						if (distances[index((*it)->data)] == UNREACHABLE || distances[index((*it)->data)] == distances[index(u)] + 1)
							mf++;
					}
					for (int i = 0; i < vertices.size(); i++)
						if (predecessors[i] == nullptr) {
							for (typename List<Vertex<T>*>::iterator it = vertices[i].adjacencyList.begin(); it != vertices[i].adjacencyList.end(); ++it) {
								mu++;
							}
						}
				}
				if (mf > mu / a && iterationComplete) return;
				else iterationComplete = true;
			}
			if (q.empty()) return;
		}
	}
	void BFSBottomUp(Vertex<T>* current, std::vector<bool>& markedVertices,
		std::vector<Vertex<T>*>& path, std::vector<int>& distances, std::vector<Vertex<T>*>& predecessors, std::vector<bool>& frontier,
		double b) {
		std::vector<bool> next(vertices.size());
		bool iterationComplete = false;
		predecessors[index(current)] = current;
		distances[index(current)] = 0;
		while (std::any_of(frontier.begin(), frontier.end(), [](bool v) {return v; })) {
			for(int i = 0; i < vertices.size(); i++)
				if (predecessors[i] == nullptr) {
					for (typename List<Vertex<T>*>::iterator it = vertices[i].adjacencyList.begin(); it != vertices[i].adjacencyList.end(); ++it) {
						if (frontier[index((*it)->data)]) {
							next[i] = true;
							predecessors[i] = (*it)->data;
							distances[i] = distances[index((*it)->data)] + 1;
							path.push_back(&(vertices[i]));
							break;
						}
					}
				}
			frontier = next;
			std::fill(next.begin(), next.end(), false);
			int nf = std::count(frontier.begin(), frontier.end(), true);
			if (nf < vertices.size() && iterationComplete) return;
			else iterationComplete = true;
		}
	}
	void DFS(Vertex<T>* current, std::vector<bool>& markedVertices, std::vector<Vertex<T>*>& path, std::vector<int>& components, int currentComponent) {
		markedVertices[index(current)] = true;
		path.push_back(current);
		components[index(current)] = currentComponent;
		for (typename List<Vertex<T>*>::iterator it = current->adjacencyList.begin(); it != current->adjacencyList.end(); ++it) {
			bool marked = false;
			if (markedVertices[index((*it)->data)]) continue;
			DFS((*it)->data, markedVertices, path, components, currentComponent);
		}
		return;
	}
	void dijkstra(Vertex<T>* current, std::vector<bool>& markedVertices, std::vector<int>& distances, std::vector<Vertex<T>*>& predecessors, std::set<VertexDistance<T>>& s) {
		
		VertexDistance<T> vd1(current, 0);
		s.insert(vd1);
		
		while (!s.empty()) {

			current = s.begin()->vertex;
			s.erase(s.begin());

			for (typename List<Vertex<T>*>::iterator it = current->adjacencyList.begin(); it != current->adjacencyList.end(); ++it) {
				if (markedVertices[index((*it)->data)]) continue;
				if (distances[index(current)] + weights[index(current)][index((*it)->data)] < distances[index((*it)->data)] || distances[index((*it)->data)] == UNREACHABLE) {
					distances[index((*it)->data)] = distances[index(current)] + weights[index(current)][index((*it)->data)];
					predecessors[index((*it)->data)] = current;
					VertexDistance<T> vd2(&(vertices[index((*it)->data)]), distances[index((*it)->data)]);
					s.insert(vd2);
				}
				markedVertices[index(current)] = true;
			}
		}
		
	}
public:
	std::vector<Vertex<T>> vertices;
	int** weights;
	Graph(std::vector<Vertex<T>> v) {
		vertices = v;
	}
	Graph(std::vector<Vertex<T>> v, int** ws) {
		vertices = v;
		weights = ws;
	}
	int index(Vertex<T>* pvx) {
		int index = pvx - &(vertices[0]);
		return index;
	}

	std::vector<Vertex<T>*> BFS(int startingVertex, std::vector<int>* d = nullptr, std::vector<Vertex<T>*>* pred = nullptr, double a = 1, double b = 1) {
		std::vector<bool> markedVertices(vertices.size());
		markedVertices[startingVertex] = true;
		std::vector<Vertex<T>*> path = {};
		path.push_back(&(vertices[startingVertex]));
		std::vector<int> distances(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = UNREACHABLE;
		}
		std::vector<Vertex<T>*> predecessors(vertices.size());
		std::queue<Vertex<T>*> q;
		q.push(&(vertices[startingVertex]));
		std::vector<bool> frontier(vertices.size());
		frontier[startingVertex] = true;
		int i = 0;
		while (!q.empty() && std::any_of(frontier.begin(), frontier.end(), [](bool v) {return v; })) {
			BFS(&(vertices[startingVertex]), markedVertices, path, distances, predecessors, q, a);
			i++;
			if (i == 2) break;
			if (!q.empty() && std::any_of(frontier.begin(), frontier.end(), [](bool v) {return v; })) {
				startingVertex = index(path[path.size() - 1]);
				//std::cout << "->";
				convertQueueToBitset(q, frontier);
				BFSBottomUp(&(vertices[startingVertex]), markedVertices, path, distances, predecessors, frontier, b);
				startingVertex = index(path[path.size() - 1]);
				//std::cout << "<-";
				convertBitsetToQueue(frontier, q);
			}
		}
		if (d != nullptr)
			*d = distances;
		if (pred != nullptr)
			*pred = predecessors;
		return path;
	}

	std::vector<Vertex<T>*> DFS(int startingVertex = 0, std::vector<int>* comp = nullptr) {
		std::vector<bool> markedVertices(vertices.size());
		std::vector<Vertex<T>*> path = {};
		std::vector<int> components(vertices.size());
		int currentComponent = 0;
		DFS(&(vertices[startingVertex]), markedVertices, path, components, currentComponent);
		currentComponent++;
		for (int i = 0; i < vertices.size(); i++) {
			if (!(markedVertices[i])) {
				DFS(&(vertices[i]), markedVertices, path, components, currentComponent);
				currentComponent++;
			}
		}
		if (comp != nullptr)
			*comp = components;
		return path;
	}
	void dijkstra(int startingVertex, std::vector<int>* d = nullptr, std::vector<Vertex<T>*>* pred = nullptr) {
		std::vector<bool> markedVertices(vertices.size());
		markedVertices[startingVertex] = true;
		std::vector<int> distances(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = UNREACHABLE;
		}
		std::vector<Vertex<T>*> predecessors(vertices.size());
		std::set<VertexDistance<T>> s;
		dijkstra(&(vertices[startingVertex]), markedVertices, distances, predecessors, s);
		if (d != nullptr)
			*d = distances;
		if (pred != nullptr)
			*pred = predecessors;
	}
	void addEdge(int v1, int v2, int weight = 0) {
		vertices[v1].adjacencyList.push_front(&(vertices[v2]));
		vertices[v2].adjacencyList.push_front(&(vertices[v1]));
		weights[v1][v2] = weight;
		weights[v2][v1] = weight;
	}
};

Graph<int> generateGraph(int nVertices, int**& weights) {
	const static int maxWeight = 5;
	std::vector<Vertex<int>> vertices(nVertices);
	weights = new int* [nVertices];
	for (int i = 0; i < nVertices; i++) {
		weights[i] = new int[nVertices];
		for (int j = 0; j < nVertices; j++)
			weights[i][j] = 0;
	}
	Graph<int> graph(vertices, weights);
	for (int i = 0; i < nVertices; i++) {
		int r = rand();
		Vertex<int> vx(Pair<int>(i, r), List<Vertex<int>*>());
		graph.vertices[i] = vx;
		int newEdges = rand() % 6;
		if (i < newEdges) newEdges = i;
		for (int j = 0; j < newEdges; j++) {
			int randVertex;
			do {
				randVertex = rand() % i;
			} while (weights[randVertex][i] != 0);
			graph.vertices[i].adjacencyList.push_front(&(graph.vertices[randVertex]));
			graph.vertices[randVertex].adjacencyList.push_front(&(graph.vertices[i]));
			int randWeight = rand() % maxWeight + 1;
			weights[i][randVertex] = randWeight;
			weights[randVertex][i] = randWeight;
		}
	}
	return graph;
}

Graph<int> washington(int N, int**& weights) {
	const int nVertices = 3 * N + 3;
	std::vector<Vertex<int>> vertices(nVertices);
	weights = new int* [nVertices];
	for (int i = 0; i < nVertices; i++) {
		weights[i] = new int[nVertices];
		for (int j = 0; j < nVertices; j++)
			weights[i][j] = 0;
	}
	
	Graph<int> graph(vertices, weights);
	for (int i = 0; i < 3 * N + 3; i++) {
		Vertex<int> vx(Pair<int>(i, N), List<Vertex<int>*>());
		graph.vertices[i] = vx;
	}
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
const int ITERATIONS = 1000;
const int NVERTICES = ITERATIONS * 9;
//const int NVERTICES = 10;

const double a = 0.0001, b = 100;

int main()
{
	//srand(time(NULL));
	
	std::vector<Vertex<int>> vertices;
	
	for (int i = 0; i < NVERTICES; i++) {
		vertices.push_back(Vertex<int>(Pair<int>(i, i * 12), List<Vertex<int>*>()));
	}
	
	int** weights = new int* [NVERTICES];
	for (int i = 0; i < NVERTICES; i++)
		weights[i] = new int[NVERTICES];
	
	for (int k = 0; k < ITERATIONS; k++) {
		weights[k * 9 + 0][k * 9 + 1] = 1;
		weights[k * 9 + 0][k * 9 + 8] = 1;
		weights[k * 9 + 1][k * 9 + 2] = 1;
		weights[k * 9 + 1][k * 9 + 3] = 3;
		weights[k * 9 + 2][k * 9 + 4] = 2;
		weights[k * 9 + 2][k * 9 + 6] = 4;
		weights[k * 9 + 2][k * 9 + 7] = 1;
		weights[k * 9 + 3][k * 9 + 5] = 6;
		weights[k * 9 + 3][k * 9 + 6] = 1;
		weights[k * 9 + 4][k * 9 + 6] = 1;
		weights[k * 9 + 5][k * 9 + 6] = 2;
		for (int i = 0; i < 9; i++)
			for (int j = i + 1; j < 9; j++) {
				weights[k * 9 + j][k * 9 + i] = weights[k * 9 + i][k * 9 + j];
			}
	}
	
	Graph<int> graph(vertices, weights);
	for (int k = 0; k < ITERATIONS; k++) {
		graph.vertices[k * 9 + 0].adjacencyList.push_front(&(graph.vertices[k * 9 + 1])); //   8-0 7 17-9  16
	//	graph.vertices[k * 9 + 0].adjacencyList.push_front(&(graph.vertices[k * 9 + 7])); //     | |    |  |
		graph.vertices[k * 9 + 0].adjacencyList.push_front(&(graph.vertices[k * 9 + 8])); //   3-1-2-12-10-11- ... 
		graph.vertices[k * 9 + 1].adjacencyList.push_front(&(graph.vertices[k * 9 + 0])); //   |\ /|  |\  /|
		graph.vertices[k * 9 + 1].adjacencyList.push_front(&(graph.vertices[k * 9 + 2])); //   5-6-4-14-15-13
		graph.vertices[k * 9 + 1].adjacencyList.push_front(&(graph.vertices[k * 9 + 3]));
		graph.vertices[k * 9 + 2].adjacencyList.push_front(&(graph.vertices[k * 9 + 1]));
		graph.vertices[k * 9 + 2].adjacencyList.push_front(&(graph.vertices[k * 9 + 4]));
		graph.vertices[k * 9 + 2].adjacencyList.push_front(&(graph.vertices[k * 9 + 6]));
		graph.vertices[k * 9 + 2].adjacencyList.push_front(&(graph.vertices[k * 9 + 7]));
		graph.vertices[k * 9 + 3].adjacencyList.push_front(&(graph.vertices[k * 9 + 1]));
		graph.vertices[k * 9 + 3].adjacencyList.push_front(&(graph.vertices[k * 9 + 5]));
		graph.vertices[k * 9 + 3].adjacencyList.push_front(&(graph.vertices[k * 9 + 6]));
		//	graph.vertices[k * 9 + 3].adjacencyList.push_front(&(graph.vertices[k * 9 + 8]));
		graph.vertices[k * 9 + 4].adjacencyList.push_front(&(graph.vertices[k * 9 + 2]));
		graph.vertices[k * 9 + 4].adjacencyList.push_front(&(graph.vertices[k * 9 + 6]));
		graph.vertices[k * 9 + 5].adjacencyList.push_front(&(graph.vertices[k * 9 + 3]));
		graph.vertices[k * 9 + 5].adjacencyList.push_front(&(graph.vertices[k * 9 + 6]));
		graph.vertices[k * 9 + 6].adjacencyList.push_front(&(graph.vertices[k * 9 + 2]));
		graph.vertices[k * 9 + 6].adjacencyList.push_front(&(graph.vertices[k * 9 + 3]));
		graph.vertices[k * 9 + 6].adjacencyList.push_front(&(graph.vertices[k * 9 + 4]));
		graph.vertices[k * 9 + 6].adjacencyList.push_front(&(graph.vertices[k * 9 + 5]));
		//	graph.vertices[k * 9 + 7].adjacencyList.push_front(&(graph.vertices[k * 9 + 0]));
		graph.vertices[k * 9 + 7].adjacencyList.push_front(&(graph.vertices[k * 9 + 2]));
		graph.vertices[k * 9 + 8].adjacencyList.push_front(&(graph.vertices[k * 9 + 0]));
		if (k != 0) {
			graph.vertices[k * 9 + 3].adjacencyList.push_front(&(graph.vertices[(k - 1) * 9 + 2]));
			graph.vertices[(k - 1) * 9 + 2].adjacencyList.push_front(&(graph.vertices[k * 9 + 3]));
			weights[k * 9 + 3][(k - 1) * 9 + 2] = 1;
			weights[(k - 1) * 9 + 2][k * 9 + 3] = 1;
			graph.vertices[k * 9 + 5].adjacencyList.push_front(&(graph.vertices[(k - 1) * 9 + 4]));
			graph.vertices[(k - 1) * 9 + 4].adjacencyList.push_front(&(graph.vertices[k * 9 + 5]));
			weights[k * 9 + 5][(k - 1) * 9 + 4] = 1;
			weights[(k - 1) * 9 + 4][k * 9 + 5] = 1;
		}
	}
	//	graph.vertices[k * 9 + 9].adjacencyList.push_front(&(graph.vertices[k * 9 + 3]));
	
	//int** weights = nullptr;
	//Graph<int> graph1 = generateGraph(NVERTICES, weights);
	Graph<int> graph1 = washington(3000, weights);

	/*
	std::vector<int> components;
	std::vector<Vertex<int>*> path = graph.DFS(0, &components);
	for (int i = 0; i < path.size(); i++)
		std::cout << path[i]->data.first << " ";
	std::cout << std::endl << std::endl;
	for (int i = 0; i < graph.vertices.size(); i++)
		std::cout << components[i] << " ";
	std::cout << std::endl << std::endl;
	*/
	std::vector<int> distances;
	std::vector<Vertex<int>*> predecessors;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	std::vector<Vertex<int>*> BFSpath = graph1.BFS(0, &distances, &predecessors, a, b);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	std::cout << time_span.count() << " seconds";
	std::cout << std::endl;
	 /*
	for (int i = 0; i < BFSpath.size(); i++)
		std::cout << BFSpath[i]->data.first << " ";/*
	std::cout << std::endl << std::endl;
	for (int i = 0; i < graph.vertices.size(); i++)
		std::cout << distances[i] << " ";
	std::cout << std::endl << std::endl;

	graph.dijkstra(1, &distances);
	for (int i = 0; i < graph.vertices.size(); i++)
		std::cout << distances[i] << " ";
	
	for (int i = 0; i < NVERTICES; i++)
		delete[] weights[i];
	delete[] weights;
	*/
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

/*
vertices[0].adjacencyList.push_front(&(vertices[2]));
	vertices[0].adjacencyList.push_front(&(vertices[3]));
	vertices[1].adjacencyList.push_front(&(vertices[2]));
	vertices[1].adjacencyList.push_front(&(vertices[3]));
	vertices[2].adjacencyList.push_front(&(vertices[1]));
	vertices[2].adjacencyList.push_front(&(vertices[4]));
	vertices[2].adjacencyList.push_front(&(vertices[0]));
	vertices[3].adjacencyList.push_front(&(vertices[1]));
	vertices[3].adjacencyList.push_front(&(vertices[0]));
	vertices[4].adjacencyList.push_front(&(vertices[2]));
	*/