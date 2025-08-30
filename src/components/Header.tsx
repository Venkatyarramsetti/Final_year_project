import { Button } from "@/components/ui/button";

const Header = () => {
  return (
    <header className="relative z-50 bg-background/80 backdrop-blur-md border-b border-border">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-primary rounded-lg flex items-center justify-center shadow-glow">
            <svg className="w-6 h-6 text-primary-foreground" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 9V7L15 1L9 7V9H21ZM12 7.5C11.2 7.5 10.5 8.2 10.5 9S11.2 10.5 12 10.5 13.5 9.8 13.5 9 12.8 7.5 12 7.5ZM6 10V12H4V10H6ZM20 10V12H18V10H20ZM6 14V16H4V14H6ZM20 14V16H18V14H20ZM6 18V20H4V18H6ZM20 18V20H18V18H20Z"/>
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">HazardSpotter</h1>
            <p className="text-xs text-muted-foreground">AI Safety Detection</p>
          </div>
        </div>
        
        <nav className="hidden md:flex items-center space-x-8">
          <a href="#home" className="text-foreground hover:text-primary transition-colors">Home</a>
          <a href="#about" className="text-foreground hover:text-primary transition-colors">About</a>
          <a href="#features" className="text-foreground hover:text-primary transition-colors">Features</a>
        </nav>
        
        <div className="flex items-center space-x-3">
          <Button variant="ghost" size="sm">Login</Button>
          <Button size="sm" className="bg-gradient-primary shadow-primary hover:shadow-glow transition-all duration-300">
            Register
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;