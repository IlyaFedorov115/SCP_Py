use [CoverTable]


DECLARE 
@table1 VARCHAR(80), 
@table2 VARCHAR(80),  
@resultCost INT,
@popSize INT,
@numIter INT,
@vectorDist VARCHAR(50)

SET @table1 = 'Cover_Table'    -- название таблицы источника таблицы, формат: Row INT, Col INT
SET @table2 = 'CoverCostsDict' -- название таблицы источника стоимостей, формат: Ind INT (номер), Cost INT
SET @vectorDist = 'euclid'     -- euclid / manhattan / cheb (варианты векторных расстояний)
SET @popSize = 40
SET @numIter = 250

EXECUTE BH_Algorithm @table1 , @table2, @resultCost OUTPUT, @popSize, @numIter, @vectorDist;


PRINT @resultCost;
SELECT * FROM BH_Algo_Result;
