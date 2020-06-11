#ifndef __DLNET_HPP__
#define __DLNET_HPP__

#include <vector>
#include <iostream>

namespace dlnet{

//======================================================================================================================
// virtual interface 
class DLNet{
	protected:
		std::vector<double> *W;
		std::vector<double> I;
		std::vector<double> O;
		std::vector<double> gradOI;  // OxI matrix
		std::vector<double> gradOW;  // OxW matrix
		int dimI;
		int dimO;
		int dimW;
												
		DLNet(int dimI,int dimW,int dimO){
			this->dimI = dimI;
			this->dimW = dimW;
			this->dimO = dimO;
		}//endDLNet

		void clearIO(){
			I.clear();
			O.clear();
		}//end_clearIO 														
		void clearGrad(){
			gradOI.clear();
			gradOW.clear();
		}//end_clearGrad				

		//===========================================================================================================						
		
		//virtual void initW() = 0; 		
		//virtual std::vector<double> operator(const std::vector<double>& I) = 0; 	// O = f(W,I)
		//virtual std::vector<double> updateGradOW() = 0; 							// \frac{\partial O}{\partial W}
		//virtual std::vector<double> updateGradOI() = 0; 							// \frac{\partial O}{\partial I}
		~DLNet(){
			delete W;
		}// end ~DLNet

};
//=========================================================================================================================



/*
class FullLinear:public DLNet{
	FullLinear(int dimI,int dimO){
		DLNet(dimI,dimI*dimO,dimO);
	}
	
	void initW(){
		W = new std::vector<double>(dimW);
		for(int i=0;i<dimW;i++){
			(*W)[i] = 0.0;
		}//endfor
	}//initW
	

};
*/


};


#endif