import { Button } from "@/components/ui/button";
import { Upload, Shield, Users, Eye } from "lucide-react";
import { useNavigate } from "react-router-dom";
import heroImage from "@/assets/hero-image.jpg";

const HeroSection = () => {
  const navigate = useNavigate();

  const handleStartDetection = () => {
    const isAuthenticated = !!localStorage.getItem("token");
    if (isAuthenticated) {
      navigate("/detection");
    } else {
      navigate("/login");
    }
  };

  return (
    <section id="home" className="relative min-h-screen bg-gradient-hero overflow-hidden">
      {/* Background Image with Overlay */}
      <div className="absolute inset-0">
        <img 
          src={heroImage} 
          alt="AI Safety Detection Technology" 
          className="w-full h-full object-cover opacity-20"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-transparent to-accent/20"></div>
      </div>
      
      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 py-24 flex items-center min-h-screen">
        <div className="max-w-3xl">
          <div className="mb-6">
            <div className="inline-flex items-center px-4 py-2 bg-card/80 backdrop-blur-sm rounded-full border border-border shadow-card">
              <Shield className="w-4 h-4 text-primary mr-2" />
              <span className="text-sm font-medium text-foreground">AI-Powered Safety Detection</span>
            </div>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold text-foreground mb-6 leading-tight">
            Detect Hazards with
            <span className="bg-gradient-primary bg-clip-text text-transparent"> AI Precision</span>
          </h1>
          
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl leading-relaxed">
            Upload images and documents to instantly identify potential hazards and safety risks. 
            Get age-specific safety assessments for teenagers, adults, and elders.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 mb-12">
            <Button 
              size="lg" 
              className="bg-gradient-primary shadow-primary hover:shadow-glow transition-all duration-300 transform hover:scale-105"
              onClick={handleStartDetection}
            >
              <Upload className="w-5 h-5 mr-2" />
              Start Detection
            </Button>
            <Button 
              variant="outline" 
              size="lg" 
              className="border-border bg-card/50 backdrop-blur-sm hover:bg-card/80 transition-all duration-300"
            >
              <Eye className="w-5 h-5 mr-2" />
              View Demo
            </Button>
          </div>
          
          {/* Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <div className="text-center sm:text-left">
              <div className="text-3xl font-bold text-primary mb-1">99.9%</div>
              <div className="text-sm text-muted-foreground">Detection Accuracy</div>
            </div>
            <div className="text-center sm:text-left">
              <div className="text-3xl font-bold text-primary mb-1">&lt;2s</div>
              <div className="text-sm text-muted-foreground">Processing Time</div>
            </div>
            <div className="text-center sm:text-left">
              <div className="text-3xl font-bold text-primary mb-1">50k+</div>
              <div className="text-sm text-muted-foreground">Images Analyzed</div>
            </div>
          </div>
        </div>
        
        {/* Floating Cards */}
        <div className="hidden lg:block absolute right-8 top-1/2 transform -translate-y-1/2 space-y-6">
          <div className="bg-card/80 backdrop-blur-sm p-6 rounded-2xl shadow-card border border-border">
            <Users className="w-8 h-8 text-primary mb-3" />
            <h3 className="font-semibold text-foreground mb-2">Age-Specific Analysis</h3>
            <p className="text-sm text-muted-foreground">Tailored safety assessments for different age groups</p>
          </div>
          <div className="bg-card/80 backdrop-blur-sm p-6 rounded-2xl shadow-card border border-border">
            <Shield className="w-8 h-8 text-accent mb-3" />
            <h3 className="font-semibold text-foreground mb-2">Real-time Detection</h3>
            <p className="text-sm text-muted-foreground">Instant hazard identification and classification</p>
          </div>
        </div>
      </div>
      
      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-primary rounded-full flex justify-center">
          <div className="w-1 h-3 bg-primary rounded-full mt-2 animate-pulse"></div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;