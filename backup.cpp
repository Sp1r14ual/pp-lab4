#include <iostream>
#include "mpi.h"
#include <chrono>
#include <cmath>
#include <vector>

using namespace std;

//Переменные и функции
const double a = 10e5; //Параметр уравнения
const double eps = 10e-6; //Допустимая погрешность
const int iter_max = 20;
const double phi0 = 0.0; //Начальное приближение

//Область моделирования
const vector<double> start_coordinate{-1.0, -1.0, -1.0};

const vector<double> D{2.0, 2.0, 2.0};

//Размер сетки
vector<int> N{160, 160, 160};

//Шаги сетки
const vector<double> H{D[0] / (double)(N[0] - 1), D[1] / (double)(N[1] - 1), D[2] / (double)(N[2] - 1)};

int sizeProc = 0;//Количество процессов
int rankProc = 0;//Номер процесса

//Функция φ
double phi_value(double x, double y, double z){
    return x * x + y * y + z * z;
}

//Правая часть
double ro(double x, double y, double z){
    return 6.0 - a * phi_value(x, y, z);
}

//Координаты узла
double xi(int i){
    return start_coordinate[0] + i * H[0];
}

double yj(int j){
    return start_coordinate[1] + j * H[1];
}

double zk(int k){
    return start_coordinate[2] + k * H[2];
}

double phi_next(double **phi_val, int k, int i){

    double part1 = 1.0 / (2.0 / (H[0] * H[0]) + 2.0 / (H[1] * H[1]) + 2.0 / (H[2] * H[2]) + a);

    double part2x = (phi_val[k][i + 1] + phi_val[k][i - 1])/(H[0] * H[0]);
    double part2y = (phi_val[k][i + N[1]] + phi_val[k][i - N[1]])/(H[1] * H[1]);
    double part2z = (phi_val[k + 1][i] + phi_val[k - 1][i])/(H[2] * H[2]);

    double part2 = part2x + part2y + part2z - 6.0 + a * phi_value(xi(i % N[0]), yj(i / N[0]), zk(k + rankProc * N[2]));

    return part1*part2;
}

int main(int argc, char**argv){

    double startwtime = 0.0, endwtime;

    MPI_Init(&argc, &argv);//Инициализация
    MPI_Comm_size(MPI_COMM_WORLD, &sizeProc); //Записываем количество процессов
    
    if (N[2] % sizeProc != 0){
    	cout << "ERROR:Process' number is too much or Process' number is not multiple to grid\n";
    	MPI_Abort(MPI_COMM_WORLD,-1);
    	MPI_Finalize();
    	return -1;
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rankProc); //Записываем номер текущего процесса

    int tag = 0;

    //Декомпозиция на каждый процесс
    //Используем декомпозицию на "линейке" по оси Z
    N[2] = N[2] / (double)sizeProc;

    //Массив значений
    double **phi = new double *[N[2]];
    for (int i = 0; i < N[2]; ++i){
        phi[i] = new double[N[1] * N[0]];
    }

    //Массив значений (предыдущий)
    double **phi_prev = new double *[N[2]];
    for (int i = 0; i < N[2]; ++i){
        phi_prev[i] = new double[N[1] * N[0]];
    }

    //Заполняем phi_val
    for (int k = 0; k < N[2]; ++k){
        for (int i = 0; i < N[1] * N[0]; ++i){
            if ((k == 0) || (k == N[2] - 1) || //Границы по Zk
                ((i / N[0]) == 0) || ((i / N[0]) == (N[1] - 1)) || //Границы по Yj
                ((i % N[0]) == 0) || ((i % N[0]) == (N[0] - 1))){ //Границы по Xi
                    //Граничные значения
                    phi[k][i] = phi_value(xi(i % N[0]), yj(i / N[0]), zk(k + rankProc * N[2]));
                } 
                else {//Внутренняя часть
                    phi[k][i] = phi0;
                }
        }
    }

    //Вычисляем приближение до выполнения условния
    double error = 1.0;
    int iteration = 1; //Номер итерации

    if (rankProc == 0){
        cout << "Start of calculations..."<<endl;
        cout << "Iterations info:" << endl;
    }

    startwtime = MPI_Wtime();
    while (error >= eps and iteration < iter_max)
    {

        //Запомним пердыдущие значение функций
        for (int k = 0; k < N[2]; ++k){
            for (int i = 0; i < N[1] * N[0]; ++i){
                phi_prev[k][i] = phi[k][i];
            }
        }

        //Вычисляем сеточные значения, прилегающие к границе локальной подобласти
        for (int k = 0; k < N[2]; ++k){
            for (int i = 0; i < N[1] * N[0]; ++i){
                if ((((k == 1) || (k == N[2] - 2)) && ((i / N[0]) >= 1) && ((i / N[0]) <= (N[1] - 2)) && ((i % N[0]) >= 1) && ((i % N[0]) <= (N[0] - 2))) || //Прилегающие к границе по Zk
                    ((((i / N[0]) == 1)||((i / N[0]) == (N[1] - 2))) && (k >= 1) && (k <= (N[2] - 2)) && ((i % N[0]) >= 1) && ((i % N[0]) <= (N[0] - 2))) || //Прилегающие к границе по Yj
                    ((((i % N[0]) == 1)||((i % N[0]) == (N[0] - 2))) && (k >= 1) && (k <= (N[2] - 2)) && ((i / N[0]) >= 1) && ((i / N[0]) <= (N[1] - 2)))){//Прилегающие к границе по Xi
                        phi[k][i] = phi_next(phi_prev, k, i);
                    }
            }
            
        }

        //Запускаем асинхронный обмен граничных значений
        MPI_Request req[4];
        MPI_Status sta[4];

        if (rankProc != 0){
            MPI_Isend(phi[1], N[0] * N[1], MPI_DOUBLE, rankProc - 1, tag, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(phi[0], N[0] * N[1], MPI_DOUBLE, rankProc - 1, tag, MPI_COMM_WORLD, &req[2]);
        }

        if (rankProc != sizeProc - 1){
            MPI_Isend(phi[N[2] - 2], N[0] * N[1], MPI_DOUBLE, rankProc + 1, tag, MPI_COMM_WORLD, &req[1]);
            MPI_Irecv(phi[N[2] - 1], N[0] * N[1], MPI_DOUBLE, rankProc + 1, tag, MPI_COMM_WORLD, &req[3]);
        }

        //Выполняем вычисление остальных точек подобласти
        for (int k = 0; k < N[2]; ++k){
            for (int i = 0; i < N[1] * N[0]; ++i){
                if ((k > 1) && (k < N[2] - 2) && //Остальные точки по Zk
                    ((i / N[0]) > 1) && ((i / N[0]) < (N[1] - 2)) && //Остальные точки по Yj
                    ((i % N[0]) > 1) && ((i % N[0]) < (N[0] - 2))) //Остальные точки по Xi
                    {
                        phi[k][i] = phi_next(phi_prev, k, i);
                    }
            }
            
        }

        //Ожидание завершения обменов
        if (rankProc != 0){
            MPI_Wait(&req[0], &sta[0]);
            MPI_Wait(&req[2], &sta[2]);
        }

        if (rankProc != sizeProc - 1){
            MPI_Wait(&req[1], &sta[1]);
            MPI_Wait(&req[3], &sta[3]);
        }

        double proc_error = 0.0;

        for (int k = 0; k < N[2]; ++k){
            for (int i = 0; i < N[1] * N[0]; ++i){
                double delta = fabs(phi[k][i] - phi_prev[k][i]);
                if (delta > proc_error){
                    proc_error = delta;
                }
            }
        }

        error = proc_error;
        MPI_Allreduce(&proc_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        //Результат итерации
        if (rankProc == 0){
            cout << iteration << ") max delta among all proc = " << error << endl;
        }
        ++iteration;
    }

    endwtime = MPI_Wtime();

    //Считаем истинное значение функции в точках
    for (int i = 0; i < N[2]; ++i){
    	for (int j = 0; j < N[1] * N[0]; ++j){
    		//Значения точек и функции в точке
                phi_prev[i][j] = phi_value(xi(j % N[0]), yj(j / N[0]), zk(i + rankProc * N[2]));
    	}
    }

    double max_delta = 0.0;
    //Ищем максимальную по модулю разность
    for(int i = 0; i < N[2]; ++i){
    	for (int j = 0; j < N[1] * N[0]; ++j){
            double delta = fabs(phi[i][j] - phi_prev[i][j]);
    	    if(delta > max_delta){
    	    	max_delta = delta;
    	    }
        }
    }
    //Определим дельту из всех потоков и выберем наибульшую
    error = max_delta;
    MPI_Allreduce(&max_delta, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rankProc == 0){
        cout << endl << "Max delta = "<< error << endl;
        cout << "Время работы программы:" << endwtime - startwtime << endl;
    }

    //Очистим память
    for (int i = 0; i < N[2]; ++i){
        delete[] phi[i];
        delete[] phi_prev[i];
    }
    delete[] phi;
    delete[] phi_prev;

    MPI_Finalize();
    return 0;
}