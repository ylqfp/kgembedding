
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
using namespace std;


#define pi 3.1415926535897932384626433832795

bool L1_flag=1;

//normal distribution
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
		{
			printf("%d\n", res);
            res+=x;
		}
        return res;
    }
int main(){
	for(int i=0; i<100; i++){
		int res = rand_max(100);
		printf("i=%d,rand_max=%d\n",i, res);
	}
}
