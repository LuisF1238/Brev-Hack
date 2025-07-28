'use client';

import { useState } from 'react';

export default function Home() {
  const [count, setCount] = useState(0);
  const [name, setName] = useState('');
  const [todos, setTodos] = useState<string[]>([]);
  const [newTodo, setNewTodo] = useState('');

  const addTodo = () => {
    if (newTodo.trim()) {
      setTodos([...todos, newTodo.trim()]);
      setNewTodo('');
    }
  };

  const removeTodo = (index: number) => {
    setTodos(todos.filter((_, i) => i !== index));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-8">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-800 dark:text-white mb-4">
            Welcome to My App
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            A modern Next.js application with interactive features
          </p>
        </header>

        <div className="grid md:grid-cols-2 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-white">
              Counter Demo
            </h2>
            <div className="text-center">
              <div className="text-4xl font-bold text-indigo-600 dark:text-indigo-400 mb-4">
                {count}
              </div>
              <div className="space-x-4">
                <button
                  onClick={() => setCount(count - 1)}
                  className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
                >
                  Decrease
                </button>
                <button
                  onClick={() => setCount(count + 1)}
                  className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
                >
                  Increase
                </button>
                <button
                  onClick={() => setCount(0)}
                  className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-white">
              Personal Greeting
            </h2>
            <div className="space-y-4">
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter your name"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              />
              {name && (
                <div className="text-lg text-indigo-600 dark:text-indigo-400 font-medium">
                  Hello, {name}! 👋
                </div>
              )}
            </div>
          </div>

          <div className="md:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-white">
              Todo List
            </h2>
            <div className="space-y-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newTodo}
                  onChange={(e) => setNewTodo(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addTodo()}
                  placeholder="Add a new todo"
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                />
                <button
                  onClick={addTodo}
                  className="px-4 py-2 bg-indigo-500 text-white rounded hover:bg-indigo-600 transition-colors"
                >
                  Add
                </button>
              </div>
              {todos.length > 0 && (
                <ul className="space-y-2">
                  {todos.map((todo, index) => (
                    <li
                      key={index}
                      className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-md"
                    >
                      <span className="text-gray-800 dark:text-white">{todo}</span>
                      <button
                        onClick={() => removeTodo(index)}
                        className="px-3 py-1 bg-red-500 text-white text-sm rounded hover:bg-red-600 transition-colors"
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>

        <footer className="text-center mt-12 text-gray-600 dark:text-gray-400">
          <p>Built with Next.js, React, and Tailwind CSS</p>
        </footer>
      </div>
    </div>
  );
}
