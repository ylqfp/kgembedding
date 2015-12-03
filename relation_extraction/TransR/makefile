RE = train_TransR test_TransR
all: ${RE}
train_TransR: Train_TransR.cpp
	g++ -g -ggdb Train_TransR.cpp -o Train_TransR -O2
test_TransR: Test_TransR.cpp
	g++ -g -ggdb Test_TransR.cpp -o Test_TransR -O2
clean: 
	rm -f ${RE}
