### DIY-DL C++概念

```cpp
//==============================================================================
class Network{
    private:
    	// 暫存
    	double *I;
    	double *O;
    	double *W;
    	double *gradOI;
    	double *gradOW;
    	int dimI;
    	int dimO;
    	int dimW;  	
    public:
    	Network(int dimI,int dimO,int dimW); // I,O = 資料運算流 , W = 模型權重 
    	void initW(); 						// 初始化權重
    	void clearIO(); 					// 清除暫存值
    	void clearGrad(); 					// 清除暫存梯度
       	double* operator(double *input) // O = f(W,I)
    	double* updateGradOW(); // \frac{\partial O}{\partial W}
    	double* updateGradOI(); // \frac{\partial O}{\partial I}
};
//===============================================================================







```



