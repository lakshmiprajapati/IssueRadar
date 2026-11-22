
import { useState } from "react";
import {
  Search,
  Bug,
  GitPullRequest,
  Star,
  GitFork,
  Code2,
  Activity,
  ShieldCheck,
  Loader2,
  AlertCircle,
} from "lucide-react";

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

import { analyzeRepository } from "./services/api";
import "./App.css";

const COLORS = ["#f87171", "#60a5fa", "#34d399", "#fbbf24", "#a78bfa"];

function App() {
  const [repoUrl, setRepoUrl] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async (e) => {
    e.preventDefault();

    if (!repoUrl.trim()) {
      setError("Please enter a GitHub repository URL.");
      return;
    }

    setLoading(true);
    setError("");
    setData(null);

    try {
      const result = await analyzeRepository(repoUrl.trim());
      setData(result);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Unable to analyze repository."
      );
    } finally {
      setLoading(false);
    }
  };

  const analytics = data?.analytics;
  const repository = data?.repository;

  const chartData = Object.entries(
    analytics?.category_distribution || {}
  ).map(([name, value]) => ({ name, value }));

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={23} />
          </div>
          <span>Issue<span className="brand-accent">Radar</span></span>
        </div>

        <div className="sidebar-section">WORKSPACE</div>

        <div className="nav-item active">
          <Activity size={18} />
          Dashboard
        </div>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          AI Engine Ready
          <div className="sidebar-version">Version 1.0.0</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <h1>Issue Intelligence Dashboard</h1>
            <p>AI-powered GitHub issue analysis and insights</p>
          </div>

          <a
            className="github-link"
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
          >
            <Code2 size={18} />
            GitHub
          </a>
        </header>

        <section className="search-card">
          <div className="section-heading">
            <div className="heading-icon">
              <Search size={20} />
            </div>
            <div>
              <h2>Analyze Repository</h2>
              <p>Enter a public GitHub repository to get started.</p>
            </div>
          </div>

          <form onSubmit={handleAnalyze} className="search-form">
            <input
              type="url"
              placeholder="https://github.com/owner/repository"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              required
            />

            <button type="submit" disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="spin" size={18} />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search size={18} />
                  Analyze
                </>
              )}
            </button>
          </form>

          {error && (
            <div className="error-message">
              <AlertCircle size={18} />
              {error}
            </div>
          )}
        </section>

        {!data && !loading && (
          <section className="welcome-card">
            <div className="welcome-icon">
              <ShieldCheck size={38} />
            </div>
            <h2>Understand your GitHub issues better</h2>
            <p>
              Discover issue categories, AI prediction confidence,
              and semantically similar issues using machine learning.
            </p>
            <div className="feature-list">
              <span><Bug size={16} /> Issue Classification</span>
              <span><Search size={16} /> Similar Issue Detection</span>
              <span><Activity size={16} /> Repository Analytics</span>
            </div>
          </section>
        )}

        {loading && (
          <div className="loading-card">
            <Loader2 className="spin" size={36} />
            <h3>Analyzing repository...</h3>
            <p>Fetching GitHub issues and running AI models.</p>
            <p className="loading-note">
              This may take a little time depending on the repository.
            </p>
          </div>
        )}

        {data && repository && (
          <>
            <section className="repo-card">
              <div className="repo-title">
                <Code2 size={25} />
                <div>
                  <h2>{repository.full_name}</h2>
                  <p>{repository.description || "No description available"}</p>
                </div>
              </div>

              <div className="repo-meta">
                <span><Star size={16} /> {repository.stars.toLocaleString()}</span>
                <span><GitFork size={16} /> {repository.forks.toLocaleString()}</span>
                <span><Code2 size={16} /> {repository.language || "Unknown"}</span>
                <a href={repository.html_url} target="_blank" rel="noreferrer">
                  View Repository ↗
                </a>
              </div>
            </section>

            <section className="stats-grid">
              <StatCard
                icon={<Bug />}
                label="Open Issues Fetched"
                value={analytics.total_issues}
                color="red"
              />

              <StatCard
                icon={<ShieldCheck />}
                label="Issues Classified"
                value={analytics.classified_issues}
                color="green"
              />

              <StatCard
                icon={<Activity />}
                label="Average Confidence"
                value={`${(analytics.average_confidence * 100).toFixed(1)}%`}
                color="blue"
              />

              <StatCard
                icon={<GitPullRequest />}
                label="Repository Open Issues"
                value={repository.open_issues_count}
                color="purple"
              />
            </section>

            <section className="content-grid">
              <div className="panel">
                <h3>Issue Category Distribution</h3>

                {chartData.length > 0 ? (
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height={260}>
                      <PieChart>
                        <Pie
                          data={chartData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={65}
                          outerRadius={100}
                          paddingAngle={4}
                        >
                          {chartData.map((entry, index) => (
                            <Cell
                              key={entry.name}
                              fill={COLORS[index % COLORS.length]}
                            />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            background: "#171c2b",
                            border: "1px solid #30394d",
                            borderRadius: "10px",
                            color: "#fff",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="empty-text">No classified issues found.</p>
                )}

                <div className="legend">
                  {chartData.map((item, index) => (
                    <div className="legend-item" key={item.name}>
                      <span
                        className="legend-dot"
                        style={{
                          background: COLORS[index % COLORS.length],
                        }}
                      />
                      <span>{item.name.replace("_", " ")}</span>
                      <strong>{item.value}</strong>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel insight-panel">
                <h3>AI Analysis Summary</h3>
                <div className="insight-item">
                  <span>Issues fetched</span>
                  <strong>{data.issues_fetched}</strong>
                </div>
                <div className="insight-item">
                  <span>Successful predictions</span>
                  <strong>{data.issues_classified}</strong>
                </div>
                <div className="insight-item">
                  <span>Average confidence</span>
                  <strong>
                    {(analytics.average_confidence * 100).toFixed(2)}%
                  </strong>
                </div>
                <div className="insight-note">
                  Predictions are generated using the trained
                  machine learning classifier.
                </div>
              </div>
            </section>

            <section className="panel issues-panel">
              <div className="issues-heading">
                <div>
                  <h3>Classified Issues</h3>
                  <p>AI-generated categories and similar issue matches</p>
                </div>
                <span className="issue-count">
                  {data.issues.length} Issues
                </span>
              </div>

              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Issue</th>
                      <th>Predicted Category</th>
                      <th>Confidence</th>
                      <th>Similar Issues</th>
                    </tr>
                  </thead>

                  <tbody>
                    {data.issues.map((issue) => {
                      const prediction = issue.prediction;

                      return (
                        <tr key={issue.id}>
                          <td>
                            <a
                              href={issue.html_url}
                              target="_blank"
                              rel="noreferrer"
                              className="issue-title"
                            >
                              #{issue.number} {issue.title}
                            </a>
                          </td>

                          <td>
                            {prediction ? (
                              <span className="category-badge">
                                {prediction.predicted_category}
                              </span>
                            ) : (
                              <span className="muted">Not classified</span>
                            )}
                          </td>

                          <td>
                            {prediction
                              ? `${(prediction.confidence * 100).toFixed(1)}%`
                              : "—"}
                          </td>

                          <td>
                            <span className="similar-count">
                              {issue.duplicate_count ?? 0}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="panel similar-panel">
              <h3>Similar Issues</h3>
              <p>
                Semantically similar issues found by the AI search system.
                Similarity does not necessarily mean confirmed duplicates.
              </p>

              {data.issues.filter(
                (issue) => issue.duplicate_issues?.length > 0
              ).length === 0 ? (
                <p className="empty-text">No similar issues found.</p>
              ) : (
                data.issues
                  .filter((issue) => issue.duplicate_issues?.length > 0)
                  .map((issue) => (
                    <div className="similar-group" key={issue.id}>
                      <strong>
                        #{issue.number} {issue.title}
                      </strong>

                      {issue.duplicate_issues.map((match, index) => (
                        <a
                          key={index}
                          href={match.html_url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {match.title || match.html_url}
                        </a>
                      ))}
                    </div>
                  ))
              )}
            </section>
          </>
        )}

        <footer>
          IssueRadar · AI-powered GitHub Issue Intelligence
        </footer>
      </main>
    </div>
  );
}

function StatCard({ icon, label, value, color }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${color}`}>{icon}</div>
      <div>
        <p>{label}</p>
        <h2>{value ?? 0}</h2>
      </div>
    </div>
  );
}

export default App;
