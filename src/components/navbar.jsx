import { useState, useEffect } from "react";
import "./Navbar.css";

function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
      <div className="logo">ChurnLabs</div>

      <ul className="nav-links">
        <li><a href="#home">Home</a></li>
        <li><a href="#predict">Predict</a></li>
        <li><a href="#dashboard">Dashboard</a></li>
      </ul>

      
    </nav>
  );
}

export default Navbar;