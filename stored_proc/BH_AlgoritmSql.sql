﻿USE [CoverTable]
GO

IF OBJECT_ID ( 'BH_Algorithm', 'P' ) IS NOT NULL
    DROP PROCEDURE BH_Algorithm;
	PRINT 'DELETE';
GO


CREATE PROCEDURE BH_Algorithm  (
		@tableName VARCHAR(80),					-- название таблицы с покрытием (например "Cover_Table")
		@costsName VARCHAR(80),					-- название таблицы Стоимостей
		@resultCost INT OUTPUT,					-- для получения результирующей стоимости
		@popSize INT,							-- параметры популяции и числа итераций
		@numIter INT,
		@vectorDist VARCHAR(20) = 'euclid',
		@notationType VARCHAR(5) = 'CS',
		@transferFun VARCHAR(5) = 'S1'

)
AS


DECLARE @requestTable NVARCHAR(max) -- Задаем запрос на таблицу покрытия со значениями стоимостей


SET @requestTable = 'SELECT ['+@tableName+'].Row, ['+@tableName+'].Col, ['+@costsName+'].Cost 
	FROM ['+@tableName+'] LEFT JOIN ['+@costsName+'] ON ['+@tableName+'].Col = ['+@costsName+'].Ind';
SET @resultCost = 0;


BEGIN
	-- создать таблицу результат (просто набор номеров вошедших рядов)
	IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'BH_Algo_Result')
	BEGIN
		PRINT 'NOT EXISTS'
		Create Table BH_Algo_Result (Row_ INT);
	END
	TRUNCATE TABLE BH_Algo_Result;


	-- применение самого алгоритма. Само решение запишется в таблицу BH_Algo_Result
	-- а результирующая стоимость выведена на экран и возвращена через OUTPUT переменную @resultCost
	INSERT INTO BH_Algo_Result 
	EXECUTE sp_execute_external_script @language = N'Python',
	@script = N'
import pandas as pd
import numpy as np

def generate_solution(size):
    return np.random.randint(0, 2, size)

def calc_fitness(solution, costs):
    dim = 0 if solution.ndim == 1 else 1
    return np.array(np.sum(np.multiply(solution, costs), dim), dtype="float32")

def standard_discrete(transfer, x):
    x = transfer(x)
    r = np.random.sample(x.shape)
    return np.where(x >= r, 1.0, 0.0)

def binarization(transfer, discretization):
    def decorator(x):
        return np.array([discretization(transfer, e) for e in x], dtype=float)
    return decorator

def vector_dist(x1, x2, norm_type="euclid"):
    if norm_type.lower() == "euclid":
        ord_ = 2
    elif norm_type.lower() == "manhattan":
        ord_ = 1
    elif norm_type.lower() == "chebyshev" or norm_type.lower() == "cheb":
        ord_ = np.Inf
    else:
        raise Exception(f"Incorrect value for parameter vector!")
    return np.linalg.norm(x2 - x1, ord=ord_)

def get_transfer_function(transfer_fun="s1"):
    if transfer_fun.lower() == "s1":
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-2.0 * x))
    elif transfer_fun.lower() == "s2":
        transfer = lambda x: 1.0 / (1.0 + np.e ** -x)
    elif transfer_fun.lower() == "s3":
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-x / 2.0))
    elif transfer_fun.lower() == "s4":
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-x / 3.0))
    elif transfer_fun.lower() == "s5":
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-3.0*x))
    elif transfer_fun.lower() == "stan":
        transfer = lambda x: np.abs(2 / np.pi * np.arctan(x * np.pi / 2))
    else:
        raise Exception(f"Incorrect value {transfer_fun} for parameter transfer_fun!")
    return transfer

def repair_solution(table, costs, notation):
    rows = {elem[0] for elem in table}
    columns = {elem[1] for elem in table}
    alpha = {f: s for f, s in zip(
        rows, [{elem[1] for elem in table if elem[0] == t} for t in rows]
    )}
    betta = {f: s for f, s in zip(
        columns, [{elem[0] for elem in table if elem[1] == t} for t in columns]
    )}

    if notation.lower() == "sc":
        alpha, betta = betta, alpha
        rows = columns

    def wrapped(solution):
        S = {i + 1 for i, e in enumerate(solution) if e == 1.0}
        w_num = {e1: e2 for e1, e2 in zip(
            rows, [len(S & alpha[i]) for i in rows]
        )}
        U = {e for e in w_num if w_num[e] == 0}
        while U:
            row = U.pop()
            j = min(alpha[row],
                    key=lambda r: np.Inf if len(U & betta[r]) == 0
                    else costs[r - 1] / len(U & betta[r]))
            S.add(j)
            for curr in betta[j]: w_num[curr] += 1
            U = U - betta[j]

        S = list(reversed(list(S)))
        for row in S[:]:
            for curr in betta[row]:
                if w_num[curr] < 2:
                    break
            else:
                S.remove(row)
                for c in betta[row]: w_num[c] -= 1
        S = np.array(S) - 1
        solution[:] = np.zeros(len(solution))
        solution[S] = 1.0
    return wrapped

def calc_rotation(stars, black_hole):
    return stars + np.random.sample(stars.shape) * (np.tile(black_hole, (stars.shape[0], 1)) - stars)

def process_collapse(stars, stars_fitness, black_hole, bh_fitness, event_horizon):
    event_radius = bh_fitness / np.sum(stars_fitness)
    if event_horizon.lower() == "standard":
        indexes = [i for i in range(len(stars))
                   if standard_dist(stars_fitness[i], bh_fitness) < event_radius]
    else:
        indexes = [i for i in range(len(stars))
                   if vector_dist(stars[i], black_hole, norm_type=event_horizon) < event_radius]

    for index in indexes:
        stars[index][:] = generate_solution(len(stars[0]))

def engine_bha(costs, pop_size, max_iter, binarization, repair_star, event_horizon):
    stars = generate_solution((pop_size, len(costs)))
    _ = list(map(repair_star, stars))
    black_hole = np.zeros(len(costs))
    bh_fitness = np.Inf

    for step in range(max_iter):
        stars_fitness = calc_fitness(stars, costs)
        min_fit_index = np.argmin(stars_fitness)

        if bh_fitness > stars_fitness[min_fit_index]:
            tmp = stars[min_fit_index].copy()
            stars[min_fit_index][:] = black_hole[:]
            black_hole[:] = tmp[:]
            bh_fitness, stars_fitness[min_fit_index] = stars_fitness[min_fit_index], bh_fitness

        stars = calc_rotation(stars, black_hole)
        stars = binarization(stars)
        process_collapse(stars, stars_fitness, black_hole, bh_fitness, event_horizon)

        _ = list(map(repair_star, stars))
    return black_hole, bh_fitness

print(pandas.__version__)
import time
start_time = time.time()

# set params
notation=notationType
transfer_fun=transferFun
discrete_fun="standard"
event_horizon= vectorDist
pop_size=popSize
max_iter=numIter

columns = np.unique(InputDataSet["Col"])
tmp_lst = InputDataSet["Col"].tolist()
costs = np.array(InputDataSet["Cost"][[tmp_lst.index(e) for e in columns]])
table = np.array(InputDataSet[["Row", "Col"]])#.to_numpy()

transfer = get_transfer_function(transfer_fun)
discrete = standard_discrete
binary_fun = binarization(transfer, discrete)
repair_star = repair_solution(table, costs, notation)
optimum, values = engine_bha(costs, pop_size, max_iter, binary_fun, repair_star, event_horizon)
print(f"Elapsed time {time.time() - start_time}")
print(f"Result cost: {values}")
optimum = pd.DataFrame(np.where(optimum > 0.0)[0], columns=["Row_"])
resultCost = int(values)
OutputDataSet = optimum
	'
	, @input_data_1 = @requestTable
	, @params = N'@resultCost INT OUTPUT, @popSize INT, @numIter INT, @vectorDist VARCHAR(20), @notationType VARCHAR(5), @transferFun VARCHAR(5)'
	, @resultCost = @resultCost OUTPUT
	, @popSize = @popSize
	, @numIter = @numIter
	, @vectorDist = @vectorDist
	, @notationType = @notationType 
	, @transferFun =  @transferFun 
	--WITH RESULT SETS(([Row_] INT NOT NULL));

END;
