@echo off
cd "c:\Users\venka\Documents\ssvenkat\final_year_project\hazard-spotter-ai"
echo Current directory: %CD% > git_debug.log
echo ------------------------------ >> git_debug.log
echo Checking if .git directory exists: >> git_debug.log
if exist .git (
    echo .git directory exists >> git_debug.log
) else (
    echo .git directory does NOT exist >> git_debug.log
    echo Initializing git repository... >> git_debug.log
    git init >> git_debug.log 2>&1
)

echo ------------------------------ >> git_debug.log
echo Git status: >> git_debug.log
git status >> git_debug.log 2>&1

echo ------------------------------ >> git_debug.log
echo Git remote configuration: >> git_debug.log
git remote -v >> git_debug.log 2>&1

echo ------------------------------ >> git_debug.log
echo Adding all files to git: >> git_debug.log
git add . >> git_debug.log 2>&1

echo ------------------------------ >> git_debug.log
echo Committing changes: >> git_debug.log
git commit -m "Enhance backend for Render deployment with better error handling and resilience" >> git_debug.log 2>&1

echo ------------------------------ >> git_debug.log
echo Setting up remote if needed: >> git_debug.log
git remote add origin https://github.com/Venkatyarramsetti/hazard-spotter-ai.git >> git_debug.log 2>&1

echo ------------------------------ >> git_debug.log
echo Pushing to remote repository: >> git_debug.log
git push -u origin main >> git_debug.log 2>&1

echo ------------------------------ >> git_debug.log
echo Git push completed. Check git_debug.log for details. >> git_debug.log