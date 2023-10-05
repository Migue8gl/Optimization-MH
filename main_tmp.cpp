#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <set>
#include <iomanip>
using namespace std;
using namespace std::chrono;

vector<vector<double>> data_matrix;
vector<char> class_vector;
random_device r;
// double Seed = r();
double Seed;

void readData(string file)
{
    vector<vector<string>> data_matrix_aux;
    string ifilename = file;
    ifstream ifile;
    istream *input = &ifile;

    ifile.open(ifilename.c_str());

    if (!ifile)
    {
        cerr << "[ERROR]Couldn't open the file" << endl;
        cerr << "[Ex.] Are you sure you are in the correct path?" << endl;
        exit(1);
    }

    string data;
    int cont = 0, cont_aux = 0;
    char aux;
    vector<string> aux_vector;
    bool finish = false;

    // Leo número de atributos y lo guardo en contador
    do
    {
        *input >> data;
        if (data == "@attribute")
            cont++;
    } while (data != "@data"); // A partir de aquí leemos datos

    data = "";

    // Mientras no lleguemos al final leemos datos
    while (!(*input).eof())
    {
        // Leemos caracter a caracter
        *input >> aux;

        /* Si hemos terminado una linea de datos la guardamos en la matrix de datos
        y reiniciamos el contador auxiliar (nos dice por qué dato vamos) */
        if (finish)
        {
            data_matrix_aux.push_back(aux_vector);
            aux_vector.clear();
            cont_aux = 0;
            finish = false;
        }

        /* Si hay una coma el dato ha terminado de leerse y lo almacenamos, en caso
        contrario seguimos leyendo caracteres y almacenandolos en data*/
        if (aux != ',' && cont_aux < cont)
        {
            data += aux;
            // Si hemos llegado al penultimo elemento hemos terminado
            if (cont_aux == cont - 1)
            {
                cont_aux++;
                aux_vector.push_back(data);
                data = "";
                finish = true;
            }
        }
        else
        {
            aux_vector.push_back(data);
            data = "";
            cont_aux++;
        }
    }

    vector<double> vect_aux;

    for (vector<vector<string>>::iterator it = data_matrix_aux.begin(); it != data_matrix_aux.end(); it++)
    {
        vect_aux.clear();
        for (vector<string>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            if (jt == it->end() - 1)
                class_vector.push_back((*jt)[0]);
            else
                vect_aux.push_back(stod(*jt));
        }
        data_matrix.push_back(vect_aux);
    }
}

void normalizeData(vector<vector<double>> &data)
{
    double item = 0.0;           // Característica individual
    double max_item = -999999.0; // Valor máximo del rango de valores
    double min_item = 999999.0;  // Valor minimo del rango de valores

    // Buscamos los máximos y mínimos
    for (vector<vector<double>>::iterator it = data.begin(); it != data.end(); it++)
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            item = *jt;

            if (item > max_item)
                max_item = item;

            if (item < min_item)
                min_item = item;
        }

    // Normalizamos aplicando x_iN = (x_i - min) / (max - min)
    for (vector<vector<double>>::iterator it = data.begin(); it != data.end(); it++)
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
            *jt = (*jt - min_item) / (max_item - min_item);
}

pair<vector<vector<vector<double>>>, vector<vector<char>>> createPartitions()
{
    vector<vector<double>> data_m_aux = data_matrix;
    vector<char> class_v_aux = class_vector;

    // Mezclo aleatoriamente la matriz original
    srand(Seed);
    random_shuffle(begin(data_m_aux), end(data_m_aux));
    srand(Seed);
    random_shuffle(begin(class_vector), end(class_vector));

    const int MATRIX_SIZE = data_matrix.size();
    vector<vector<double>>::iterator it = data_m_aux.begin();
    vector<char>::iterator jt = class_v_aux.begin();

    // Particiones y puntero que las irá recorriendolas para insertar datos
    vector<vector<double>> g1, g2, g3, g4, g5, *g_aux;
    vector<char> g1c, g2c, g3c, g4c, g5c, *g_aux2;
    int cont = 0, cont_grupos = 0;
    bool salir = false;

    // Mientras no se hayan insertado todos los datos en todos los grupos
    while (cont != MATRIX_SIZE && cont_grupos < 5)
    {
        // Elegimos la partición que toque
        switch (cont_grupos)
        {
        case 0:
            g_aux = &g1;
            g_aux2 = &g1c;
            break;
        case 1:
            g_aux = &g2;
            g_aux2 = &g2c;
            break;
        case 2:
            g_aux = &g3;
            g_aux2 = &g3c;
            break;
        case 3:
            g_aux = &g4;
            g_aux2 = &g4c;
            break;
        case 4:
            g_aux = &g5;
            g_aux2 = &g5c;
            break;
        }

        // Vamos rellenando la partición pertinente
        for (int k = 0; k < MATRIX_SIZE / 5 && !salir; k++)
        {
            g_aux->push_back(*it);
            g_aux2->push_back(*jt);
            it++;
            jt++;
            cont++;

            /* Si estamos en el último grupo y quedan todavía elementos, seguir
            insertándolos en este último */
            if (cont_grupos == 4)
            {
                if (it != data_m_aux.end())
                    k--;
                else
                    salir = true;
            }
        }
        cont_grupos++;
    }
    vector<vector<vector<double>>> d = {g1, g2, g3, g4, g5};
    vector<vector<char>> c = {g1c, g2c, g3c, g4c, g5c};
    pair<vector<vector<vector<double>>>, vector<vector<char>>> partitions = make_pair(d, c);

    return partitions;
}

char KNN_Classifier(vector<vector<double>> &data, vector<vector<double>>::iterator &elem, vector<char> &elemClass, vector<double> &w)
{
    vector<double> distancia;
    vector<char> clases;
    vector<char>::iterator cl = elemClass.begin();
    vector<double>::iterator wi = w.begin();
    vector<double>::iterator ej;
    double sumatoria = 0;
    double dist_e = 0;

    for (vector<vector<double>>::iterator e = data.begin(); e != data.end(); e++)
    {
        // Si el elemento es él mismo no calculamos distancia, pues es 0
        if (elem != e)
        {
            sumatoria = 0;
            ej = elem->begin();
            wi = w.begin();

            // Calculamos distancia de nuestro elemento con el resto
            for (vector<double>::iterator ei = e->begin(); ei != e->end(); ei++)
            {
                sumatoria += *wi * pow(*ej - *ei, 2);
                ej++;
                wi++;
            }
            dist_e = sqrt(sumatoria);
            distancia.push_back(dist_e);
            clases.push_back(*cl);
        }
        cl++;
    }

    vector<double>::iterator it;
    vector<char>::iterator cl_dist_min = clases.begin();

    double distMin = 99999;
    char vecinoMasProxClass;

    // Nos quedamos con el que tenga minima distancia, es decir, su vecino más próximo
    for (it = distancia.begin(); it != distancia.end(); it++)
    {
        if (*it < distMin)
        {
            distMin = *it;
            vecinoMasProxClass = *cl_dist_min;
        }
        cl_dist_min++;
    }

    return vecinoMasProxClass;
}

double calculaAciertos(vector<vector<double>> &muestras, vector<char> &clases, vector<double> &w)
{
    double instBienClasificadas = 0.0;
    double numIntanciasTotal = float(muestras.size());
    char cl_1NN;
    vector<char>::iterator c_it = clases.begin();

    for (vector<vector<double>>::iterator it = muestras.begin(); it != muestras.end(); it++)
    {
        cl_1NN = KNN_Classifier(muestras, it, clases, w);

        if (cl_1NN == *c_it)
            instBienClasificadas += 1.0;
        c_it++;
    }

    return instBienClasificadas / numIntanciasTotal;
}

void execute(pair<vector<vector<vector<double>>>, vector<vector<char>>> &part, vector<double> (*alg)(vector<vector<double>> &, vector<char> &, string), string arg)
{
    vector<double> w;
    vector<vector<vector<double>>>::iterator data_test = part.first.begin();
    vector<vector<char>>::iterator class_test = part.second.begin();
    vector<vector<double>> aux_data_fold;
    vector<char> aux_class_fold;
    vector<vector<vector<double>>>::iterator it;
    vector<vector<char>>::iterator jt;

    double tasa_clas = 0;
    double tasa_red = 0;
    double agregado = 0;
    double alpha = 0.5;
    unsigned int cont_red = 0;
    double TS_media = 0, TR_media = 0, A_media = 0;
    int cont = 0;

    auto momentoInicio = high_resolution_clock::now();

    // Iteramos 5 veces ejecutando el algoritmo
    while (cont < 5)
    {
        jt = part.second.begin();
        aux_data_fold.clear();
        aux_class_fold.clear();
        cont_red = 0;

        // Creamos particiones train
        for (it = part.first.begin(); it != part.first.end(); it++)
        {
            // Si es una partición test no la añadimos a training
            if (it != data_test && jt != class_test)
            {
                aux_data_fold.insert(aux_data_fold.end(), (*it).begin(), (*it).end());
                aux_class_fold.insert(aux_class_fold.end(), (*jt).begin(), (*jt).end());
            }
            jt++;
        }

        // Ejecución del algoritmo
        auto partInicio = high_resolution_clock::now();
        w = alg(aux_data_fold, aux_class_fold, arg);
        auto partFin = high_resolution_clock::now();

        cont_red = 0;
        for (vector<double>::iterator wi = w.begin(); wi != w.end(); wi++)
        {
            if (*wi < 0.1)
            {
                cont_red += 1;
                *wi = 0.0;
            }
        }

        tasa_clas = calculaAciertos(*data_test, *class_test, w);
        tasa_red = float(cont_red) / float(w.size());
        agregado = alpha * tasa_clas + (1 - alpha) * tasa_red;

        milliseconds tiempo_part = duration_cast<std::chrono::milliseconds>(partFin - partInicio);

        cout << "[PART " << cont + 1 << "] | Tasa_clas: " << tasa_clas << endl;
        cout << "[PART " << cont + 1 << "] | Tasa_red: " << tasa_red << endl;
        cout << "[PART " << cont + 1 << "] | Fitness: " << agregado << endl;
        cout << "[PART " << cont + 1 << "] | Tiempo_ejecucion: " << tiempo_part.count() << " ms\n\n";
        cout << "-------------------------------------------\n"
             << endl;

        TS_media += tasa_clas;
        TR_media += tasa_red;
        A_media += agregado;

        cont++;
        data_test++;
        class_test++;
    }
    auto momentoFin = high_resolution_clock::now();

    milliseconds tiempo = duration_cast<std::chrono::milliseconds>(momentoFin - momentoInicio);

    cout << "***** (RESULTADOS FINALES) *****\n"
         << endl;
    cout << "Tasa_clas_media: " << TS_media / 5.0 << endl;
    cout << "Tasa_red_media: " << TR_media / 5.0 << endl;
    cout << "Fitness_medio: " << A_media / 5.0 << endl;
    cout << "Tiempo_ejecucion_medio: " << tiempo.count() << " ms";
}

vector<double> evalua(vector<vector<double>> &muestra, vector<char> &muestra_clases, vector<vector<double>> &poblacion)
{
    double tasa_clas;
    double tasa_red;
    unsigned int cont_reducc;
    vector<double> f;
    f.resize(poblacion.size());
    int i = 0;

    for (vector<vector<double>>::iterator it = poblacion.begin(); it != poblacion.end(); it++)
    {
        cont_reducc = 0;
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            if (*jt < 0.1)
            {
                cont_reducc += 1;
                *jt = 0;
            }
        }
        tasa_clas = calculaAciertos(muestra, muestra_clases, *it);
        tasa_red = float(cont_reducc) / float((*it).size());
        f[i] = (tasa_red + tasa_clas) * 0.5;
        i++;
    }

    return f;
}

double evalua2(vector<vector<double>> &muestra, vector<char> &muestra_clases, vector<double> &poblacion)
{
    double tasa_clas = 0;
    double tasa_red = 0;
    double cont_reducc = 0;
    double f;

    for (vector<double>::iterator it = poblacion.begin(); it != poblacion.end(); it++)
    {
        if (*it < 0.1)
        {
            cont_reducc += 1.0;
            *it = 0;
        }
    }
    tasa_clas = calculaAciertos(muestra, muestra_clases, poblacion);
    tasa_red = float(cont_reducc) / float(poblacion.size());
    f = tasa_red * 0.5 + tasa_clas * 0.5;

    return f;
}

vector<double> torneoBinario(vector<vector<double>> &poblacion, vector<double> &eval)
{
    random_device r;
    int index1 = r() % poblacion.size();
    int index2 = r() % poblacion.size();

    while (index1 == index2)
        index2 = r() % poblacion.size();

    if (eval[index1] >= eval[index2])
        return poblacion.at(index1);
    else
        return poblacion.at(index2);
}

vector<vector<double>> cruceBLX(vector<double> &cromosoma1, vector<double> &cromosoma2, double varianza)
{
    vector<vector<double>> hijos;
    vector<double> d1, d2;
    double cmin = 999999, cmax = -999999;

    hijos.resize(2);
    d1.resize(cromosoma1.size());
    d2.resize(cromosoma2.size());
    mt19937 eng(Seed);
    mt19937 eng2(Seed);

    for (int i = 0; i < cromosoma1.size(); i++)
    {
        if (cromosoma1[i] < cromosoma2[i])
        {
            cmin = cromosoma1[i];
            cmax = cromosoma2[i];
        }
        else
        {
            cmin = cromosoma2[i];
            cmax = cromosoma1[i];
        }

        double I = cmax - cmin;

        uniform_real_distribution<double> dist(cmin - I * varianza, cmax + I * varianza);

        d1[i] = dist(eng);
        d2[i] = dist(eng2);
    }

    hijos[0] = d1;
    hijos[1] = d2;

    return hijos;
}

vector<vector<double>> cruceArimetico(vector<double> &cromosoma1, vector<double> &cromosoma2)
{
    vector<vector<double>> hijos;
    vector<double> d1, d2;

    mt19937 eng(Seed);
    mt19937 eng2(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    hijos.resize(2);
    d1.resize(cromosoma1.size());
    d2.resize(cromosoma2.size());

    double alpha = dist(eng);
    double alpha2 = dist(eng2);

    for (int i = 0; i < d1.size(); i++)
    {

        d1[i] = alpha * cromosoma1[i] + (1 - alpha) * cromosoma2[i];
        d2[i] = alpha2 * cromosoma2[i] + (1 - alpha2) * cromosoma1[i];
    }

    hijos[0] = d1;
    hijos[1] = d2;

    return hijos;
}

vector<double> algAGG(vector<vector<double>> &muestra, vector<char> &muestra_clases, string cruce)
{
    double p_cruce = 0.7;
    double p_mutacion = 0.1;
    int maxIter = 15000;
    int cont = 0, contindexjt = 0, contindexit = 0;
    int num_mut = 0, num_cruces = 0;
    double varianza = 0.3, alpha = 0.5;
    int index_gen, index_crom;
    double eval_pob_max, eval_hij_min;
    vector<double> evaluacion_hijos;
    vector<vector<double>> padres, hijos;
    vector<double>::iterator jt;
    // double s = r();
    int index_jt = 0, index_it = 0;

    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    // Inizializo población con dist. uniforme
    int num_individuos = 30;
    vector<vector<double>> poblacion;
    vector<double> elem(muestra.begin()->size());

    poblacion.resize(num_individuos);
    for (int i = 0; i < num_individuos; i++)
    {
        generate(begin(elem), end(elem), gen);
        poblacion[i] = elem;
    }

    vector<double> evaluacion = evalua(muestra, muestra_clases, poblacion);
    cont += num_individuos;

    // Creo vector z de mutacion y un generador de distribución normal
    num_mut = p_mutacion * poblacion.size() * poblacion.begin()->size();
    vector<double> z(num_mut);
    vector<double>::iterator z_it;
    normal_distribution<double> normal_dist(0.0, sqrt(varianza));
    mt19937 other_eng(Seed);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    while (cont < maxIter)
    {
        // Iniciada la iteración t borro los antiguos valores de padres e hijos
        padres.clear();
        hijos.clear();

        if (padres.size() != num_individuos)
            padres.resize(num_individuos);
        if (hijos.size() != num_individuos)
            hijos.resize(num_individuos);

        // Creo nuevos padres basándome en el torneo binario
        for (int i = 0; i < num_individuos; i++)
            padres[i] = torneoBinario(poblacion, evaluacion);

        num_cruces = floor((p_cruce * num_individuos) / 2.0);
        vector<vector<double>> aux;
        int j = 0;

        // Comienzo cruces entre padres
        if (cruce == "BLX")
        {
            for (int i = 0; i < num_cruces && i < num_individuos; i++)
            {
                // Le pasamos dos padres, siendo cromosoma1 y cromosoma2
                aux = cruceBLX(padres[i], padres[i + 1], varianza);
                hijos[j] = aux[0];
                j++;
                hijos[j] = aux[1];
                j++;
                i++;
            }
        }
        else
        {
            for (int i = 0; i < num_cruces && i < num_individuos; i++)
            {
                // Le pasamos dos padres, siendo cromosoma1 y cromosoma2
                aux = cruceArimetico(padres[i], padres[i + 1]);
                hijos[j] = aux[0];
                j++;
                hijos[j] = aux[1];
                j++;
                i++;
            }
        }

        // Añado los padres no cruzados
        for (int i = num_cruces; i < num_individuos; i++)
        {
            hijos[j] = padres[i];
            j++;
        }

        // Comienzo las mutaciones
        generate(begin(z), end(z), genNormalDist);
        for (int i = 0; i < num_mut; i++)
        {
            index_gen = r() % poblacion.begin()->size();
            index_crom = r() % poblacion.size();

            hijos[index_crom][index_gen] += z[i];

            // Hace falta?? Tu función reducira igualmetne pero bueno..
            if (hijos[index_crom][index_gen] < 0.0)
                hijos[index_crom][index_gen] = 0;
            else if (hijos[index_crom][index_gen] > 1)
                hijos[index_crom][index_gen] = 1;
        }

        evaluacion_hijos = evalua(muestra, muestra_clases, hijos);
        cont += num_individuos;

        eval_pob_max = -99999.0;
        eval_hij_min = 99999.0;
        jt = evaluacion_hijos.begin();
        contindexjt = 0, contindexit = 0;

        // Busco con elitismo al mejor candidato de la población
        for (vector<double>::iterator it = evaluacion.begin(); it != evaluacion.end(); it++)
        {
            if (*it > eval_pob_max)
            {
                eval_pob_max = *it;
                index_it = contindexit;
            }

            if (*jt < eval_hij_min)
            {
                eval_hij_min = *jt;
                index_jt = contindexjt;
            }
            jt++;
            contindexjt++;
            contindexit++;
        }

        // Una vez encontrado, lo sustituyo en los hijos por el peor candidato de estos
        *(hijos.begin() + index_jt) = poblacion[index_it];
        *(evaluacion_hijos.begin() + index_jt) = eval_pob_max;

        // Actualizo la población
        poblacion.swap(hijos);
        evaluacion.swap(evaluacion_hijos);
    }

    eval_pob_max = -99999.0;
    contindexit = 0;
    // Busco al mejor cromosoma de la población, el que mejor función objetivo tenga
    for (vector<double>::iterator it = evaluacion.begin(); it != evaluacion.end(); it++)
    {
        if (*it > eval_pob_max)
        {
            eval_pob_max = *it;
            index_it = contindexit;
        }
        contindexit++;
    }

    return poblacion[index_it];
}

vector<double> algAGE(vector<vector<double>> &muestra, vector<char> &muestra_clases, string cruce)
{
    double p_cruce = 0.7;
    double p_mutacion = 0.1;
    int maxIter = 15000;
    int cont = 0, contindexit = 0;
    int num_mut = 0, num_cruces = 0;
    double varianza = 0.3, alpha = 0.5;
    int index_gen, index_crom;
    double prob;
    double eval_pob_min, eval_pob_max;
    vector<double> evaluacion_hijos;
    vector<vector<double>> padres, hijos;
    vector<double>::iterator jt;
    int index_it = 0;
    // double s = r();

    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    // Inizializo población con dist. uniforme
    int num_individuos = 30;
    vector<vector<double>> poblacion;
    vector<double> elem(muestra.begin()->size());

    poblacion.resize(num_individuos);
    for (int i = 0; i < num_individuos; i++)
    {
        generate(begin(elem), end(elem), gen);
        poblacion[i] = elem;
    }

    vector<double> evaluacion = evalua(muestra, muestra_clases, poblacion);
    cont += num_individuos;

    // Creo vector z de mutacion y un generador de distribución normal
    vector<double> z(2);
    vector<double>::iterator z_it;
    normal_distribution<double> normal_dist(0.0, sqrt(varianza));
    mt19937 other_eng(Seed);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    while (cont < maxIter)
    {
        // Iniciada la iteración t borro los antiguos valores de padres e hijos
        padres.clear();
        hijos.clear();

        if (padres.size() != 2)
            padres.resize(2);
        if (hijos.size() != 2)
            hijos.resize(2);

        // Creo nuevos padres basándome en el torneo binario
        for (int i = 0; i < 2; i++)
            padres[i] = torneoBinario(poblacion, evaluacion);

        vector<vector<double>> aux;
        int j = 0;

        // Comienzo cruces entre padres
        if (cruce == "BLX")
        {
            // Le pasamos dos padres, siendo cromosoma1 y cromosoma2
            aux = cruceBLX(padres[0], padres[1], varianza);
            hijos[0] = aux[0];
            hijos[1] = aux[1];
        }
        else
        {
            // Le pasamos dos padres, siendo cromosoma1 y cromosoma2
            aux = cruceArimetico(padres[0], padres[1]);
            hijos[0] = aux[0];
            hijos[1] = aux[1];
        }

        // Comienzo las mutaciones
        generate(begin(z), end(z), genNormalDist);

        for (int i = 0; i < 2; i++)
        {
            if (r() % 1 < p_mutacion)
            {
                index_gen = r() % poblacion.begin()->size();
                index_crom = r() % 2;

                hijos[index_crom][index_gen] += z[i];

                if (hijos[index_crom][index_gen] < 0.0)
                    hijos[index_crom][index_gen] = 0;
                else if (hijos[index_crom][index_gen] > 1)
                    hijos[index_crom][index_gen] = 1;
            }
        }

        evaluacion_hijos = evalua(muestra, muestra_clases, hijos);
        cont += 2;

        bool h1 = false, h2 = false;

        // Reemplazo los peores de la población por las mutaciones si estas son mejores
        for (int i = 0; i < 2; i++)
        {
            eval_pob_min = 9999.0;
            contindexit = 0;

            for (vector<double>::iterator it = evaluacion.begin(); it != evaluacion.end(); it++)
            {
                if (*it < eval_pob_min)
                {
                    eval_pob_min = *it;
                    index_it = contindexit;
                }
                contindexit++;
            }
            if (evaluacion_hijos[0] > eval_pob_min && !h1)
            {
                poblacion[index_it] = hijos[0];
                evaluacion[index_it] = evaluacion_hijos[0];
                h1 = true;
            }
            else if (evaluacion_hijos[1] > eval_pob_min && !h2)
            {
                poblacion[index_it] = hijos[1];
                evaluacion[index_it] = evaluacion_hijos[1];
                h2 = true;
            }
        }
    }

    eval_pob_max = -99999.0;
    contindexit = 0;
    // Busco al mejor cromosoma de la población, el que mejor función objetivo tenga
    for (vector<double>::iterator it = evaluacion.begin(); it != evaluacion.end(); it++)
    {
        if (*it > eval_pob_max)
        {
            eval_pob_max = *it;
            index_it = contindexit;
        }
        contindexit++;
    }

    return poblacion[index_it];
}

int blAlg(vector<vector<double>> &muestra, vector<char> &muestra_clases, vector<double> &w, double &nuevas_eval)
{
    // NO es constante para los genéticos ya que tienes que restarle cuantas evaluacioens ya has hecho.
    const int maxIter = 15000;
    const int maxVecin = w.size() * 2;
    int eval = 0;
    int cont = 0, vecinos = 0;
    double varianza = 0.3, alpha = 0.5;

    // Creo vector z y un generador de distribución normal
    vector<double> z(w.size());
    vector<double>::iterator z_it;
    normal_distribution<double> normal_dist(0.0, sqrt(varianza));
    double s = r();
    mt19937 other_eng(Seed);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    double tasa_clas = 0;
    double tasa_red = 0;
    double fun_objetivo = 0;
    unsigned int cont_red = 0;
    double max_fun = -99999.0;
    double w_aux;

    fun_objetivo = evalua2(muestra, muestra_clases, w);
    max_fun = fun_objetivo;
    eval++;

    // Mientras no se superen las iteraciones máximas o los vecinos permitidos
    while (vecinos < maxVecin)
    {
        generate(begin(z), end(z), genNormalDist);
        z_it = z.begin();
        cont_red = 0;

        for (vector<double>::iterator it = w.begin(); it != w.end(); it++)
        {
            // Guardamos w original
            w_aux = *it;

            // Mutación normal
            *it += *z_it;

            if (*it < 0)
                *it = 0;
            else if (*it > 1)
                *it = 1;

            if (*it < 0.1)
            {
                *it = 0;
                cont_red += 1;
            }

            fun_objetivo = evalua2(muestra, muestra_clases, w);
            eval++;

            // Si hemos mejorado el umbral a mejorar cambia, vamos maximizando la función
            if (fun_objetivo > max_fun)
            {
                max_fun = fun_objetivo;
                vecinos = 0;
            }
            else
            {
                // Si no hemos mejorado nos quedamos con la w anterior
                *it = w_aux;
                vecinos++;
            }
            z_it++;
        }
        cont++;
    }

    nuevas_eval = max_fun;
    return eval;
}

vector<double> algAM(vector<vector<double>> &muestra, vector<char> &muestra_clases, string tipo_prob)
{
    double p_cruce = 0.7;
    double p_mutacion = 0.1;
    int maxIter = 15000;
    int index;
    int cont = 0, contindexjt = 0, contindexit = 0;
    int num_mut = 0, num_cruces = 0;
    double varianza = 0.3, alpha = 0.5;
    int index_gen, index_crom;
    double eval_pob_max, eval_hij_min;
    vector<double> evaluacion_hijos;
    vector<vector<double>> padres, hijos;
    vector<double>::iterator jt;
    int it;
    set<int> pos_visitadas;
    double s = r();
    double pls = 0.1;
    int generaciones = 1;
    int index_jt = 0, index_it = 0;

    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    // Inizializo población con dist. uniforme
    int num_individuos = 10;
    vector<vector<double>> poblacion;
    vector<double> elem(muestra.begin()->size());

    poblacion.resize(num_individuos);
    for (int i = 0; i < num_individuos; i++)
    {
        generate(begin(elem), end(elem), gen);
        poblacion[i] = elem;
    }

    vector<double> evaluacion = evalua(muestra, muestra_clases, poblacion);
    cont += num_individuos;

    // Creo vector z de mutacion y un generador de distribución normal
    num_mut = p_mutacion * poblacion.size() * poblacion.begin()->size();
    vector<double> z(num_mut);
    vector<double>::iterator z_it;
    normal_distribution<double> normal_dist(0.0, sqrt(varianza));
    mt19937 other_eng(Seed);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    while (cont < maxIter)
    {
        // Iniciada la iteración t borro los antiguos valores de padres e hijos
        padres.clear();
        hijos.clear();

        if (padres.size() != num_individuos)
            padres.resize(num_individuos);
        if (hijos.size() != num_individuos)
            hijos.resize(num_individuos);

        // Creo nuevos padres basándome en el torneo binario
        for (int i = 0; i < num_individuos; i++)
            padres[i] = torneoBinario(poblacion, evaluacion);

        num_cruces = floor((p_cruce * num_individuos) / 2.0);
        vector<vector<double>> aux;
        int j = 0;

        // Comienzo cruces entre padres
        for (int i = 0; i < num_cruces && i < num_individuos; i++)
        {
            // Le pasamos dos padres, siendo cromosoma1 y cromosoma2
            aux = cruceBLX(padres[i], padres[i + 1], varianza);
            hijos[j] = aux[0];
            j++;
            hijos[j] = aux[1];
            j++;
            i++;
        }

        // Añado los padres no cruzados
        for (int i = num_cruces; i < num_individuos - 1; i++)
        {
            hijos[j] = padres[i];
            j++;
        }

        // Comienzo las mutaciones
        generate(begin(z), end(z), genNormalDist);
        for (int i = 0; i < num_mut; i++)
        {
            index_gen = r() % poblacion.begin()->size();
            index_crom = r() % poblacion.size();

            hijos[index_crom][index_gen] += z[i];

            if (hijos[index_crom][index_gen] < 0.0)
                hijos[index_crom][index_gen] = 0;
            else if (hijos[index_crom][index_gen] > 1)
                hijos[index_crom][index_gen] = 1;
        }

        evaluacion_hijos = evalua(muestra, muestra_clases, hijos);
        cont += num_individuos;

        eval_pob_max = -99999.0;
        eval_hij_min = 99999.0;
        jt = evaluacion_hijos.begin();
        contindexjt = 0, contindexit = 0;

        // Busco con elitismo al mejor candidato de la población
        for (vector<double>::iterator it = evaluacion.begin(); it != evaluacion.end(); it++)
        {
            if (*it > eval_pob_max)
            {
                eval_pob_max = *it;
                index_it = contindexit;
            }

            if (*jt < eval_hij_min)
            {
                eval_hij_min = *jt;
                index_jt = contindexjt;
            }
            jt++;
            contindexjt++;
            contindexit++;
        }

        // Una vez encontrado, lo sustituyo en los hijos por el peor candidato de estos
        *(hijos.begin() + index_jt) = poblacion.at(index_it);
        *(evaluacion_hijos.begin() + index_jt) = eval_pob_max;

        // Aplico búsqueda local cada 10 generaciones
        if (generaciones % 10 == 0)
        {
            switch (stoi(tipo_prob))
            {
            case 0:
                for (int i = 0; i < num_individuos; i++)
                {
                    it = blAlg(muestra, muestra_clases, hijos[i], evaluacion_hijos[i]);
                    cont += it;
                }
                break;
            case 1:
                for (int i = 0; i < floor(pls * num_individuos) && i < num_individuos; i++)
                {
                    index = r() % num_individuos;
                    it = blAlg(muestra, muestra_clases, hijos[index], evaluacion_hijos[index]);
                    cont += it;
                }
                break;
            case 2:
                for (int i = 0; i < floor(pls * num_individuos) && i < num_individuos; i++)
                {
                    eval_pob_max = -99999.0;
                    contindexit = 0;
                    index_it = 0;
                    for (int j = 0; j < num_individuos; j++)
                    {
                        if (evaluacion_hijos[j] > eval_pob_max)
                        {
                            if (pos_visitadas.find(j) == pos_visitadas.end())
                            {
                                eval_pob_max = evaluacion_hijos[j];
                                index_it = contindexit;
                                pos_visitadas.insert(index_it);
                            }
                        }
                        contindexit++;
                    }
                    it = blAlg(muestra, muestra_clases, hijos[index_it], evaluacion_hijos[index_it]);
                    cont += it;
                }
                pos_visitadas.clear();
                break;
            }
        }

        // Actualizo la población
        poblacion.swap(hijos);
        evaluacion.swap(evaluacion_hijos);
        generaciones++;
    }

    eval_pob_max = -99999.0;
    contindexit = 0;
    // Busco al mejor cromosoma de la población, el que mejor función objetivo tenga
    for (vector<double>::iterator it = evaluacion.begin(); it != evaluacion.end(); it++)
    {
        if (*it > eval_pob_max)
        {
            eval_pob_max = *it;
            index_it = contindexit;
        }
        contindexit++;
    }

    return poblacion[index_it];
}

int main(int nargs, char *args[])
{
    char *arg[4];
    string option;
    string path;

    if (nargs <= 2)
    {
        cerr << "[ERROR] Wrong execution pattern" << endl;
        cerr << "[Ex.] ./main {seed} [1-3] " << endl;
        cerr << "[Pd:] 1=spectf-heart, 2=parkinsons, 3=ionosphere" << endl;
    }
    Seed = atof(args[1]);
    option = args[2];

    if (option == "1")
        path = "./Instancias_APC/spectf-heart.arff";
    else if (option == "2")
        path = "./Instancias_APC/parkinsons.arff";
    else if (option == "3")
        path = "./Instancias_APC/ionosphere.arff";
    else
    {
        cerr << "[ERROR] Parámetro no reconocido..." << endl;
        cerr << "[Ex.] Tienes que definir que data-set: 1-spectf-heart, 2-parkinsons, 3-ionosphere..." << endl;
        cerr << "[Ex.] ./main {seed} [1-3] " << endl;
        exit(1);
    }

    readData(path);
    normalizeData(data_matrix);

    pair<vector<vector<vector<double>>>, vector<vector<char>>> part;
    part = createPartitions();

    srand(Seed);
    cout << "\nSemilla: " << setprecision(10) << Seed << endl;

    cout << "\n------------(ALGORITMO AGG_BLX)------------\n\n";
    execute(part, algAGG, "BLX");

    cout << "\n\n------------(ALGORITMO AGG_Aritmetico)------------\n\n";
    execute(part, algAGG, "ARIT");

    cout << "\n\n------------(ALGORITMO AGE_BLX)------------\n\n";
    execute(part, algAGE, "BLX");

    cout << "\n\n------------(ALGORITMO AGE_Aritmetico)------------\n\n";
    execute(part, algAGE, "ARIT");

    cout << "\n\n------------(ALGORITMO AM tipo 1)------------\n\n";
    execute(part, algAM, "0");

    cout << "\n\n------------(ALGORITMO AM tipo 2)------------\n\n";
    execute(part, algAM, "1");

    cout << "\n\n------------(ALGORITMO AM tipo 3)------------\n\n";
    execute(part, algAM, "2");

    cout << endl
         << endl;
}
