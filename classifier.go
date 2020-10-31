package main

import (
	"fmt"
	"html"
	"html/template"
	"log"
	"net/http"
)

func errwrap(f func(w http.ResponseWriter, r *http.Request) error) func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := f(w, r); err != nil {
			http.Error(w, fmt.Sprintf("%+v", err), http.StatusInternalServerError)
		}
	}
}

func main() {
	http.Handle("/data/", http.FileServer(http.Dir("/data")))
	http.HandleFunc("/", errwrap(func(w http.ResponseWriter, r *http.Request) error {
		t, err := template.New("foo").Parse(`{{define "T"}}Hello, {{.}}!{{end}}`)
		if err != nil {
			return err
		}
		if err := t.ExecuteTemplate(w, "T", "<script>alert('you have been pwned')</script>"); err != nil {
			return err
		}
		fmt.Fprintf(w, "Hello, %q", html.EscapeString(r.URL.Path))
		return nil
	}))

	log.Fatal(http.ListenAndServe(":8080", nil))
}
