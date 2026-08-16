#include <iostream>

class LegacyUser {
    int id;
public:
    LegacyUser(int _id) { id = _id; }
    void print() { std::cout << id << std::endl; }
};

int* create_array(int size) {
    return new int[size];
}

void free_array(int* arr) {
    delete[] arr;
}

int main() {
    LegacyUser u(42);
    u.print();

    int* arr = create_array(10);
    for (int i = 0; i < 10; i++) {
        arr[i] = i * i;
    }
    for (int i = 0; i < 10; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    free_array(arr);

    return 0;
}
