import Data.Random
import Control.Monad

rs :: Int -> IO [Double]
rs n = replicateM n ( sample ( StdNormal))

mean:: [Double] -> Double
mean xs = let(res,len) = foldl(\(m,n) x->(m + x/len, n + 1))(0, 0) xs in res

meanIO :: Int -> IO Double
meanIO n = do
  num <- rs(n)
  let mean_value = mean(num)
  return mean_value

main = do
    let loop i | i <= 1000 = do
            --print i
            print =<< meanIO(100)
            loop $ i + 1
        loop _ = return ()
    loop 1
