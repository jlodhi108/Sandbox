function greet(name) {
    return "Hello, " + name + "!";
}

var counter = 0;

function increment() {
    counter = counter + 1;
    return counter;
}

console.log(greet("world"));
console.log(increment());
console.log(increment());
