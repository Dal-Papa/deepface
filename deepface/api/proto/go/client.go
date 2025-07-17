package main

import (
	"context"
	"flag"
	"log"
	"os"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/serengil/deepface/proto"
)

var flagImg = flag.String("img", "", "Path to the image file")

func main() {
	ctx := context.Background()
	flag.Parse()
	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}
	defer conn.Close()

	f, err := os.Open(*flagImg)
	if err != nil {
		log.Fatalf("error opening image file: %v", err)
	}
	defer f.Close()
	// Read the image file into a byte slice
	imageData, err := os.ReadFile(*flagImg)
	if err != nil {
		log.Fatalf("error reading image file: %v", err)
	}

	client := pb.NewAnalyzeServiceClient(conn)
	res, err := client.Analyze(ctx, &pb.AnalyzeRequest{
		Image: imageData,
		Actions: []pb.AnalyzeRequest_Action{
			pb.AnalyzeRequest_AGE,
			pb.AnalyzeRequest_GENDER,
			pb.AnalyzeRequest_RACE,
		},
	})
	if err != nil {
		log.Fatalf("error calling Analyze: %v", err)
	}
	log.Printf("Analyze response: %+v", res)
}
