import { Button } from "@/components/ui/button";
import { Link, useNavigate } from "react-router-dom";

const Header = () => {
  const navigate = useNavigate();
  const isLoggedIn = !!localStorage.getItem("token");

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  return (
    <header className="relative z-50 bg-background/80 backdrop-blur-md border-b border-border">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-primary rounded-lg flex items-center justify-center shadow-glow">
            <svg className="w-6 h-6 text-primary-foreground" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 9V7L15 1L9 7V9H21ZM12 7.5C11.2 7.5 10.5 8.2 10.5 9S11.2 10.5 12 10.5 13.5 9.8 13.5 9 12.8 7.5 12 7.5ZM6 10V12H4V10H6ZM20 10V12H18V10H20ZM6 14V16H4V14H6ZM20 14V16H18V14H20ZM6 18V20H4V18H6ZM20 18V20H18V18H20Z"/>
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">HazardSpotter</h1>
            <p className="text-xs text-muted-foreground">AI Safety Detection</p>
          </div>
        </Link>
        
        <nav className="hidden md:flex items-center space-x-8">
          <Link to="/" className="text-foreground hover:text-primary transition-colors">Home</Link>
          <Link to="/about" className="text-foreground hover:text-primary transition-colors">About</Link>
          <Link to="/features" className="text-foreground hover:text-primary transition-colors">Features</Link>
        </nav>
        
        <div className="flex items-center space-x-3">
          {isLoggedIn ? (
            <>
              <Button variant="ghost" size="sm" onClick={handleLogout}>Logout</Button>
            </>
          ) : (
            <>
              <Button variant="ghost" size="sm" onClick={() => navigate("/login")}>Login</Button>
              <Button 
                size="sm" 
                className="bg-gradient-primary shadow-primary hover:shadow-glow transition-all duration-300"
                onClick={() => navigate("/register")}
              >
                Register
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;