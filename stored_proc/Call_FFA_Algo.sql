use [CoverTable]


DECLARE 
@table1 VARCHAR(80), 
@table2 VARCHAR(80),  
@resultCost INT,
@popSize INT,
@numIter INT,
@vectorDist VARCHAR(50),
@transferFun VARCHAR(5),
@move_type VARCHAR(20),
@gamma FLOAT,
@gamma_alter INT,
@betta_0 FLOAT,
@alpha FLOAT,
@alpha_inf FLOAT,
@alpha_0 FLOAT,
@notationType VARCHAR(5)

SET @table1 = 'Cover_Table'    -- название таблицы источника таблицы, формат: Row INT, Col INT
SET @table2 = 'CoverCostsDict' -- название таблицы источника стоимостей, формат: Ind INT (номер), Cost INT
SET @vectorDist = 'euclid'     -- euclid / manhattan / cheb (варианты векторных расстояний)
SET @popSize = 30
SET @numIter = 150
SET @transferFun = 'S1'
SET @move_type  = 'standart' -- разновидность перемещения светлячков
SET @gamma = 1.0				-- параметры алгоритма (0;1]
SET @gamma_alter = 0			-- альтернативный gamma. Если 0, то старый, иначе gamma = gamma / maxDist^gamma_alter рекомендованы [1,2,3]
SET @betta_0 = 1.0				-- параметр алгоритма, значение [0:1]
SET @alpha = 1.0				-- параметр алгоритма, значение [0;1]
SET @alpha_inf = 0.0			-- альтернативный параметр alpha
SET @alpha_0 = 0.0				-- должны задаваться оба, тогда alpha = alpha_inf + (alpha_0 - alpha_inf)*e^t (t- итерация, будет уменьшаться alpha)
SET @notationType = 'CS'		-- нотация CS - столбцы покрывают строки, SС - строки покрывают столбцы

EXECUTE FFA_Algorithm @table1, @table2, @resultCost OUTPUT, @popSize, @numIter, @vectorDist, @rowName='Row', @colName='Col', @notationType=@notationType, @transferFun=@transferFun, @move_type=@move_type, @gamma=@gamma, @gamma_alter=@gamma_alter, @betta_0=@betta_0, @alpha=@alpha, @alpha_inf=@alpha_inf, @alpha_0=@alpha_0;


PRINT @resultCost;
SELECT * FROM FFA_Algo_Result;

--SELECT DISTINCT Cover_Table_.Col FROM Cover_Table_ Where Row in (Select DISTINCT FROM BH_Algo_Result);
