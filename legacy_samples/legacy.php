<?php

function greet($name) {
    return "Hello, " . $name . "!";
}

function add($a, $b) {
    return $a + $b;
}

echo greet("world") . "\n";
echo add(2, 3) . "\n";
