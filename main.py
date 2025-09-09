import { useState } from 'react';
import './App.css';

function App() {
  const [topic, setTopic] = useState('');
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState('');
  const [showQuestionSection, setShowQuestionSection] = useState(false);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSummary('');
    setAnswer('');
    setShowQuestionSection(false);
    
    // Check if the topic is empty
    if (!topic) {
      alert("Please enter a topic to analyze.");
      setLoading(false);
      return;
    }
  
    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/analyze_topic`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic }),
      });
  
      if (response.ok) {
        const data = await response.json();
        setSummary(data.message);
        setShowQuestionSection(true);
      } else {
        const errorData = await response.json();
        alert(`Error: ${errorData.detail}`);
      }
    } catch (error) {
      alert("Failed to connect to the backend. Please check if the service is running.");
    } finally {
      setLoading(false);
    }
  };
  
  const handleAskQuestion = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAnswer('');
    
    // Check if the question is empty
    if (!question) {
      alert("Please enter a question.");
      setLoading(false);
      return;
    }
  
    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });
  
      if (response.ok) {
        const data = await response.json();
        setAnswer(data.answer);
      } else {
        const errorData = await response.json();
        alert(`Error: ${errorData.detail}`);
      }
    } catch (error) {
      alert("Failed to connect to the backend. Please check if the service is running.");
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>AI News Digest Bot</h1>
        <p>Enter a topic to get a summary and ask questions based on recent news articles.</p>
        <form onSubmit={handleAnalyze}>
          <input
            type="text"
            placeholder="e.g., 'artificial intelligence'"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            disabled={loading}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze Topic'}
          </button>
        </form>
        {summary && (
          <div className="summary-section">
            <p>{summary}</p>
          </div>
        )}
        {showQuestionSection && (
          <div className="question-section">
            <h3>Ask a question about the topic:</h3>
            <form onSubmit={handleAskQuestion}>
              <input
                type="text"
                placeholder="e.g., 'What are the recent breakthroughs?'"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={loading}
              />
              <button type="submit" disabled={loading}>
                {loading ? 'Thinking...' : 'Get Answer'}
              </button>
            </form>
            {answer && (
              <div className="answer-section">
                <p>{answer}</p>
              </div>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;